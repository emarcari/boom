/*
  Copyright (C) 2013 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

#ifndef BART_POSTERIOR_SAMPLER_BASE_HPP_
#define BART_POSTERIOR_SAMPLER_BASE_HPP_

#include <Models/Bart/Bart.hpp>
#include <Models/GaussianModel.hpp>
#include <Models/PosteriorSamplers/PosteriorSampler.hpp>

namespace BOOM {
  // This is the base class for PosteriorSampler classes for drawing
  // from concrete Bart models.  This class handles moves that modify
  // the tree structure, so that derived classes can focus on the
  // details of data augmentation (for non-Gaussian models).
  //
  // All the models share some common structure in their prior
  // distribution.  There is p(tree) * p(variable | tree) * p(cutpoint
  // | variable, tree, ancestors) * p(mean | tree).  The middle two
  // distributions are often ignored because they are uniform.
  //
  // The prior probability that a node at depth 'd' splits into
  // children at depth (d + 1) is a / (1 + d)^b.  Given a split, a
  // variable is chosen uniformly from the set of available variables,
  // and a cutpoint uniformly from the set of available cutpoints.
  // Note that 'available' is influenced by a node's position in the
  // tree, because splits made by ancestors will make some splits
  // logically impossible, and impossible splits are not 'available.'
  // For example, descendents cannot split on the same dummy variable
  // as an ancestor.
  //
  // The conditional prior on the mean parameters at the leaves is
  // N(prior_mean_guess, prior_mean_sd).  Derived classes may require
  // priors on other model components (e.g. sigma^2 for the Gassian
  // case).
  class BartPosteriorSamplerBase : public PosteriorSampler {
   public:
    BartPosteriorSamplerBase(BartModelBase *model,
                             double prior_mean_guess,
                             double prior_mean_sd,
                             double prior_tree_depth_alpha,
                             double prior_tree_depth_beta);

    // The destructor should clear pointers to the
    // ResidualRegressionData owned by this class but observed by the
    // trees in the model.
    virtual ~BartPosteriorSamplerBase();

    // I implemented a skeleton version of logpri() to get this class
    // to compile.  It does not return anything useful, and will throw
    // an exception if called.
    virtual double logpri()const;

    // The draw method includes a call to check_residuals, to ensure
    // they have been created and placed where they need to be, and
    // calls to modify tree.
    virtual void draw();

    // Returns a draw of the mean parameter for the given leaf,
    // conditional on the tree structure and the data assigned to
    // leaf.  This differs slightly across the exponential family
    // because different ways to do data augmentation.
    virtual double draw_mean(Bart::TreeNode *leaf) = 0;

    // Returns the log density of the set of Y's described by suf,
    // conditional on sigma, but integrating mu out over the prior.
    // That is,
    //
    // log  \int p(Y | \mu, \sigma) p(\mu | \sigma) d \mu
    virtual double log_integrated_likelihood(
        const Bart::SufficientStatisticsBase *suf)const = 0;

    // Clear the vector of residuals (make it empty).
    virtual void clear_residuals() = 0;

    // Returns the number of observations stored in the residual vector.
    virtual int residual_size()const = 0;

    // Creates and stores the residual observation corresponding to
    // observation i in the model.  The model needs to have data
    // assigned before this function can be called.
    virtual Bart::ResidualRegressionData * create_and_store_residual(int i) = 0;

    // Create the type of sufficient statistics that go along with the
    // type of your data.
    virtual Bart::SufficientStatisticsBase * create_suf() const = 0;

    //----------------------------------------------------------------------
    // Verify that the vector of residuals has been computed, and that
    // the trees owned by model_ are populated.  If the trees don't
    // have the right data, clear them and put insert the residual
    // data.
    void check_residuals();

    //--------------------------------------------------------------
    // Moves used to implement draw.

    // Modify the tree structure and sample the terminal means given
    // the new structure.
    void modify_tree(Bart::Tree *tree);

    // Does one MH step on the structure of 'tree', conditional on
    // sigma, but integrating over the mean parameters.
    void modify_tree_structure(Bart::Tree *tree);

    // Attempt a birth move on the specified tree using Metropolis Hastings.
    void birth_move(Bart::Tree *tree);

    // Attempt a death move on the specified tree using Metropolis Hastings.
    void death_move(Bart::Tree *tree);

    // Conditional on the tree structure and sigma, sample the mean
    // parameters at the leaves.
    void draw_terminal_means_and_adjust_residuals(Bart::Tree *tree);

    //----------------------------------------------------------------------
    // Quantities required to compute MH acceptance ratios.
    //
    // These methods are not marked const because they can involve
    // leaves of the model recomputing their sufficient statistics.

    // Returns the contribution that the specified node and its
    // children would make to log posterior if a split were to occur
    // at *node.
    double split_log_posterior(Bart::TreeNode *node);

    // Returns the contribution that the specified *node would make to
    // log posterior if it was a leaf.
    double nosplit_log_posterior(Bart::TreeNode *node);

    // Returns the log of the probability of proposing a specific split
    // from the current leaf, given that a split move is to be
    // attempted.  The proposal probability is
    //
    // p(propose birth move) * p(leaf)
    //                       * p(variable to split on | leaf)
    //                       * p(cutpoint | variable, leaf)
    //
    // The variable and cutpoint portions of the split distribution
    // cancel when computing MH acceptance ratios, so we skip them here.
    //
    // TODO(stevescott): The p(leaf) = 1.0 / #leaves is not quite right.
    // It should be 1.0 / the number of leaves with a split available.
    //
    // Args:
    //   leaf:  The node at which to consider splitting.
    //   tree:  The tree containing leaf.
    double split_proposal_log_probability(
        const Bart::TreeNode *leaf, const Bart::Tree * tree);

    // Returns the log of the probability of proposing a specific
    // pruning move at the node 'node'.
    //
    // There are two ways this function can be called.  If you're
    // considering pruning 'tree' at 'leaf', then you can simply query
    // the tree for the number of candidate nodes.
    //
    // You can also call this function to get the probability of a
    // reverse move from 'tree' with an implicit split at 'node' back to
    // 'tree'.  In that case one extra internal node with no
    // grandchildren must be added to 'tree'.
    //
    // The function determines which case we're in by examining whether
    // 'node' has children.
    //
    // Args:
    //   node: Either a leaf (in the case of a reverse-proposal) or an
    //     internal node with no grandchildren (a 'nog' node, in the
    //     case of a death-move proposal).
    //   tree:  The tree that owns 'node'.
    double prune_proposal_log_probability(
        const Bart::TreeNode *node, const Bart::Tree *tree);

    // Compute the log of the prior probability of splitting (or not
    // splitting) at the given depth.  The root is depth zero.  Its
    // children are depth 1, etc.  This is the a/(1 + d)^b probability
    // mentioned in the preamble comments.
    double log_probability_of_split(int depth) const;
    double log_probability_of_no_split(int depth) const;

    // The prior distribution for the mean parameter at each node.
    const GaussianModel * node_mean_prior()const;

   protected:
    // Removes all pointers to residuals_ from the trees owned by
    // model_.
    void clear_data_from_trees();
   private:
    BartModelBase *model_;
    // We keep alpha on both the log scale and the raw scale.
    double log_prior_tree_depth_alpha_;
    double prior_tree_depth_alpha_;
    double prior_tree_depth_beta_;
    Ptr<GaussianModel> node_mean_prior_;

    // The types of moves considered by the Metropolis-Hastings
    // algorithm.
    enum MoveType {BIRTH = 0, DEATH = 1};
    // The vector of move_probabilities_ must be the same length as
    // the number of elements in the MoveType enum.
    static const Vector move_probabilities;
    static const double log_probability_of_proposing_birth_move;
    static const double log_probability_of_proposing_death_move;
  };

}
#endif // BART_POSTERIOR_SAMPLER_BASE_HPP_
