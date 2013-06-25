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

#include <Models/Bart/ResidualRegressionData.hpp>
#include <Models/Bart/PosteriorSamplers/BartPosteriorSampler.hpp>
#include <distributions.hpp>
#include <cpputil/math_utils.hpp>

namespace {
  // Returns the log of the integer d.  This is a compiler
  // optimization to pre-compute the logs of the first 100 integers.
  inline double log_integer(const int d) {
    switch (d) {
      case 1: return log(1);
      case 2: return log(2);
      case 3: return log(3);
      case 4: return log(4);
      case 5: return log(5);
      case 6: return log(6);
      case 7: return log(7);
      case 8: return log(8);
      case 9: return log(9);
      case 10: return log(10);
      case 11: return log(11);
      case 12: return log(12);
      case 13: return log(13);
      case 14: return log(14);
      case 15: return log(15);
      case 16: return log(16);
      case 17: return log(17);
      case 18: return log(18);
      case 19: return log(19);
      case 20: return log(20);
      case 21: return log(21);
      case 22: return log(22);
      case 23: return log(23);
      case 24: return log(24);
      case 25: return log(25);
      case 26: return log(26);
      case 27: return log(27);
      case 28: return log(28);
      case 29: return log(29);
      case 30: return log(30);
      case 31: return log(31);
      case 32: return log(32);
      case 33: return log(33);
      case 34: return log(34);
      case 35: return log(35);
      case 36: return log(36);
      case 37: return log(37);
      case 38: return log(38);
      case 39: return log(39);
      case 40: return log(40);
      case 41: return log(41);
      case 42: return log(42);
      case 43: return log(43);
      case 44: return log(44);
      case 45: return log(45);
      case 46: return log(46);
      case 47: return log(47);
      case 48: return log(48);
      case 49: return log(49);
      case 50: return log(50);
      case 51: return log(51);
      case 52: return log(52);
      case 53: return log(53);
      case 54: return log(54);
      case 55: return log(55);
      case 56: return log(56);
      case 57: return log(57);
      case 58: return log(58);
      case 59: return log(59);
      case 60: return log(60);
      case 61: return log(61);
      case 62: return log(62);
      case 63: return log(63);
      case 64: return log(64);
      case 65: return log(65);
      case 66: return log(66);
      case 67: return log(67);
      case 68: return log(68);
      case 69: return log(69);
      case 70: return log(70);
      case 71: return log(71);
      case 72: return log(72);
      case 73: return log(73);
      case 74: return log(74);
      case 75: return log(75);
      case 76: return log(76);
      case 77: return log(77);
      case 78: return log(78);
      case 79: return log(79);
      case 80: return log(80);
      case 81: return log(81);
      case 82: return log(82);
      case 83: return log(83);
      case 84: return log(84);
      case 85: return log(85);
      case 86: return log(86);
      case 87: return log(87);
      case 88: return log(88);
      case 89: return log(89);
      case 90: return log(90);
      case 91: return log(91);
      case 92: return log(92);
      case 93: return log(93);
      case 94: return log(94);
      case 95: return log(95);
      case 96: return log(96);
      case 97: return log(97);
      case 98: return log(98);
      case 99: return log(99);
      case 100: return log(100);
      default: return log(d);
    }
  }
} // namespace

namespace BOOM {

  using Bart::ResidualRegressionData;
  using Bart::Tree;
  using Bart::TreeNode;
  using Bart::VariableSummary;

  const Vector BartPosteriorSamplerBase::move_probabilities(".5 .5");
  const double BartPosteriorSamplerBase::
  log_probability_of_proposing_birth_move(
      log(BartPosteriorSamplerBase::move_probabilities[0]));

  const double BartPosteriorSamplerBase::
  log_probability_of_proposing_death_move(
      log(BartPosteriorSamplerBase::move_probabilities[1]));

  BartPosteriorSamplerBase::BartPosteriorSamplerBase(
      BartModelBase *model,
      double prior_mean_guess,
      double prior_mean_sd,
      double prior_tree_depth_alpha,
      double prior_tree_depth_beta)
      : model_(model),
        log_prior_tree_depth_alpha_(log(prior_tree_depth_alpha)),
        prior_tree_depth_alpha_(prior_tree_depth_alpha),
        prior_tree_depth_beta_(prior_tree_depth_beta),
        node_mean_prior_(new GaussianModel(
            prior_mean_guess,
            square(prior_mean_sd)))
  {
    if (prior_tree_depth_alpha <= 0
        || prior_tree_depth_alpha >= 1) {
      report_error("The prior_tree_depth_alpha parameter "
                   "must be strictly between 0 and 1.");
    }
    if (prior_tree_depth_beta < 0) {
      report_error("The prior_tree_depth_beta parameter "
                   " must be non-negative");
    }
  }

  BartPosteriorSamplerBase::~BartPosteriorSamplerBase() {
    //    clear_data_from_trees();

    // I would like to call clear_data_from_trees, so that when the
    // posterior sampler gets destroyed, another one can be used in
    // its place.  However, if this gets destroyed because the Ptr
    // that owns it is killed by the model's destructor, then the
    // trees that own the data are in an undetermined state (at least
    // undetermined by me).  In that case calling
    // clear_data_from_trees() can lead to a crash.
    //
    // It seems to be a better idea to leave the data in the trees
    // (which will point to deallocated memory when *this is
    // destoyed).
  }

  double BartPosteriorSamplerBase::logpri() const {
    report_error("logpri() is not yet implemented for "
                 "BartPosteriorSamplerBase, and it probably won't "
                 "be any time soon.");
    return -1;
  }

  void BartPosteriorSamplerBase::draw() {
    check_residuals();
    for (int i = 0; i < model_->number_of_trees(); ++i) {
      modify_tree(model_->tree(i));
    }
  }

  void BartPosteriorSamplerBase::check_residuals() {
    if (residual_size() != model_->sample_size()) {
      clear_residuals();
      clear_data_from_trees();
      for (int i = 0; i < model_->sample_size(); ++i) {
        Bart::ResidualRegressionData *data = create_and_store_residual(i);
        for (int j = 0; j < model_->number_of_trees(); ++j) {
          model_->tree(j)->populate_data(data);
        }
      }
      for (int i = 0; i < model_->number_of_trees(); ++i) {
        model_->tree(i)->populate_sufficient_statistics(create_suf());
      }
    }
  }

  void BartPosteriorSamplerBase::modify_tree(Tree *tree) {
    tree->remove_mean_effect();
    modify_tree_structure(tree);
    draw_terminal_means_and_adjust_residuals(tree);
  }

  void BartPosteriorSamplerBase::modify_tree_structure(Tree *tree) {
    MoveType move = MoveType(rmulti_mt(rng(), move_probabilities));
    switch (move) {
      case BIRTH:
        birth_move(tree);
        break;
      case DEATH:
        death_move(tree);
        break;
      default:
        report_error("An impossible move type was attempted in "
                     "BartPosteriorSamplerBase::modify_tree_structure");
    }
  }

  void BartPosteriorSamplerBase::birth_move(Tree *tree) {
    // Choose a leaf uniformly at random, from among the leaves with
    // at least 5 observations.
    //
    // Select a variable at random from the set of available
    // variables.  Select a cutpoint at random from the set of
    // available cutpoints.
    //
    // Note that as you go deeper in the tree, some variables can
    // become impossible to split on because one side of the split
    // would necessarily be empty.  For example, you would not split
    // on a dummy variable if an ancestor split on the same variable.
    // Likewise, if an ancestor split on variable 1 at cutpoint 3.2,
    // and you're on the right hand path from that ancestor, you can't
    // split on a value of variable 1 less than 3.2.
    //
    // The acceptance probability for the MH proposal used in this
    // move is derived below.
    /*
      \documentclass{article}
      \usepackage{amsmath}
      \begin{document}

      The MH accpetance probability for a BART birth move from tree $T$ to a
      candidate tree $T^*$ is $\min(\alpha, 1)$ where
      \begin{equation*}
      \alpha = \frac{p(T^*)}{p(T)}
      \frac{q(T^* \rightarrow T)}{q(T \rightarrow T^*)}.
      \end{equation*}

      The proposal $q(T \rightarrow T^*)$ involves choosing a random leaf
      $\ell$, choosing a variable $v$ to split on, from among the set of
      variables with available splits not already used by ancestors of
      $\ell$, and choosing a random cutpoint for $v$ from the set of
      remaining cutpoints not already used (or made logically impossible) by
      ancestors of $\ell$.  Thus, if we let $p(B)$ denote the probability of
      attempting a birth move,
      \begin{equation*}
      q(T \rightarrow T^*) = p(B) p(\ell | T) p(v | \ell) p(c | v, \ell).
      \end{equation*}

      The reverse move $q(T^* \rightarrow T)$ involves choosing a node $n$
      from among the interior nodes of $T^*$ with no grandchildren, so if
      $p(D)$ is the probability of attempting a death move
      \begin{equation*}
      q(T^* \rightarrow T) = p(D) p(n | T^*).
      \end{equation*}

      The last part of $\alpha$ is the ratio $p(T^*) / p(T)$.  Let $\ell$
      denote the leaf in $T$ where the split to $T^*$ takes place, and let
      $ch(\ell)$ denote the children of $\ell$ in $T^*$.  Let $d$ be the
      depth of $\ell$, so that $d+1$ is the depth of $ch(\ell)$, and let
      $s_d$ denote the probability of a split at depth $d$.  Let $R(\ell)$
      denote the data assigned to node $\ell$.

      \begin{equation*}
      \frac{p(T^*)}{p(T)} = \frac{s_d (1-s_{d+1})^2}{1-s_d}
      p(v | \ell)
      p(c | v, \ell)
      \frac{p(R(ch(\ell)))}{p(R(\ell))}.
      \end{equation*}

      \end{document}
    */

    // Step 1: Select a leaf at random from the set of available
    // leaves in the tree.
    TreeNode *leaf = NULL;
    bool node_can_split = false;
    while (node_can_split == false) {
      // Select a node and a variable uniformly at random.  Select a
      // cutpoint uniformly at random from the set of cutpoints
      // available to that node for that variable.
      leaf = tree->random_leaf(rng());

      // variable_index is the proposal for which variable should be
      // used in the new splitting rule.
      int variable_index = random_int_mt(
          rng(), 0, model_->number_of_variables() - 1);
      const VariableSummary &variable_summary(
          model_->variable_summary(variable_index));

      double cutpoint = 0;
      node_can_split = variable_summary.random_cutpoint(
          BOOM::GlobalRng::rng,
          leaf,
          &cutpoint);

      if (node_can_split) {
        // You can set a leaf's cutpoint and index safely, because
        // they are only used if the split is accepted.
        leaf->set_variable_and_cutpoint(
            variable_index,
            cutpoint);
      }
    }

    double log_proposal_probability =
        split_proposal_log_probability(leaf, tree);
    double log_reverse_proposal_probablity =
        prune_proposal_log_probability(leaf, tree);
    double candidate_log_posterior = split_log_posterior(leaf);
    double current_log_posterior = nosplit_log_posterior(leaf);

    double log_alpha_numerator =
        candidate_log_posterior - log_proposal_probability;
    double log_alpha_denominator =
        current_log_posterior - log_reverse_proposal_probablity;

    double log_alpha = log_alpha_numerator - log_alpha_denominator;
    double logu = log(runif_mt(rng()));
    if (logu < log_alpha) {
      tree->grow(leaf);  // accept the proposal
    } else {
      // Do nothing.
    }
  }

  //----------------------------------------------------------------------
  void BartPosteriorSamplerBase::death_move(Tree *tree) {
    TreeNode *candidate_leaf = tree->random_parent_of_leaves(rng());
    if (!candidate_leaf) {
      // No such node is available.  Reject this move.
      return;
    }

    double log_proposal_probability =
        prune_proposal_log_probability(candidate_leaf, tree);
    double log_reverse_proposal_probablity =
        split_proposal_log_probability(candidate_leaf, tree);
    double candidate_log_posterior = nosplit_log_posterior(candidate_leaf);
    double current_log_posterior = split_log_posterior(candidate_leaf);

    double log_alpha = (candidate_log_posterior - log_proposal_probability)
        - (current_log_posterior - log_reverse_proposal_probablity);

    if (log(runif_mt(rng())) < log_alpha) {
      tree->prune_descendants(candidate_leaf);
    } else {
      // Do nothing.  This block is here in case we want to do some
      // logging.
    }
  }

  //----------------------------------------------------------------------
  void BartPosteriorSamplerBase::draw_terminal_means_and_adjust_residuals(
      Bart::Tree *tree) {
    for(Tree::NodeSetIterator it = tree->leaf_begin();
        it != tree->leaf_end(); ++it) {
      Bart::TreeNode *leaf = *it;
      double mean = draw_mean(leaf);
      leaf->set_mean(mean);
      leaf->replace_mean_effect();
    }
  }

  //----------------------------------------------------------------------
  // Returns the log posterior of the tree that would obtain if a
  // split were to take place at 'leaf' (whether or not 'leaf'
  // currently has any children).  A variable and cutpoint must have
  // been assigned to 'leaf' before calling this function.
  //
  // The posterior is:
  //   p(split here) * p(no split below)^2
  //                 * p(variable) * p(cutpoint)  <-- implicit
  //                 * p(left data) * p(right data)
  //
  // The variable and cutpoint probabilities will cancel with the
  // proposal distribution (one direction or the other) in the MH
  // acceptance probability, so they are not computed explicitly.
  double BartPosteriorSamplerBase::split_log_posterior(
      TreeNode *leaf) {
    // First, fill the sufficient statistics that would be computed if
    // we split on leaf.
    const std::vector<ResidualRegressionData *> &node_data(leaf->data());
    boost::shared_ptr<Bart::SufficientStatisticsBase>
        left_suf(create_suf());
    boost::shared_ptr<Bart::SufficientStatisticsBase>
        right_suf(create_suf());
    int variable_index = leaf->variable_index();
    double cutpoint = leaf->cutpoint();
    for (int i = 0;  i < node_data.size(); ++i) {
      ResidualRegressionData *dp(node_data[i]);
      bool left = dp->x()[variable_index] < cutpoint;
      Bart::SufficientStatisticsBase *suf =
          left ? left_suf.get() : right_suf.get();
      suf->update(*dp);
    }

    // Compute the log of 'p(split here) * p(no split below)^2'.
    int depth = leaf->depth();
    double log_split_prob =
        log_probability_of_split(depth)
        + 2 * log_probability_of_no_split(depth + 1);

    // The contributions of the data on the left and right are
    // conditionally independent.
    double log_data_prob = log_integrated_likelihood(left_suf.get())
        + log_integrated_likelihood(right_suf.get());

    return log_split_prob + log_data_prob;
  }

  //----------------------------------------------------------------------
  // Returns the log of the posterior probability of the tree at the
  // given node, calculated as-if the node were a leaf.  Note that
  // 'leaf' may actually have children, but it will be implicitly
  // treated as a leaf by this function.
  double BartPosteriorSamplerBase::nosplit_log_posterior(TreeNode *leaf) {
    // The posterior is the prior (which is proportional to the probability
    // of not splitting at the leaf), times the likelihood, which is the
    // probability of the data at the leaf
    //
    // p(no split) * p(data)
    //             * p(variable | node)
    //             * p(cutpoint | variable, node)
    //
    // Note that p(variable | node) * p(cutpoint | variable, node)
    // will cancel in the MH proposal, so their contribution is not
    // computed.
    return log_probability_of_no_split(leaf->depth())
        + log_integrated_likelihood(leaf->compute_suf());
  }

  //----------------------------------------------------------------------
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
  double BartPosteriorSamplerBase::split_proposal_log_probability(
      const TreeNode *leaf, const Tree *tree) {
    int number_of_leaves = tree->number_of_leaves();
    if (!leaf->is_leaf()) {
      // In this case, the function is being called as part of a death
      // move.  The 'leaf' is really a parent whose children might be
      // pruned, and this function is being called to get the
      // probability of the reverse proposal _back_ to tree.  The
      // number of leaves we should be considering is the number after
      // pruning the children of 'leaf.'  That's number_of_leaves() -
      // 2 + 1, where -2 is because of the two lost children, and +1
      // because 'leaf' will become a leaf.
      --number_of_leaves;
    }
    return log_probability_of_proposing_birth_move
        - log_integer(number_of_leaves);
  }

  //----------------------------------------------------------------------
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
  //   node: Either a leaf (in the case of a reverse-proposal) or a
  //     parent of leaves (in the case of a death-move proposal).
  //   tree:  The tree that owns 'node'.
  double BartPosteriorSamplerBase::prune_proposal_log_probability(
      const TreeNode *node,
      const Tree *tree) {
    // A 'parent of leaves' is an interior node with no grandchildren.
    int number_of_parents_of_leaves = tree->number_of_parents_of_leaves();

    if (node->is_leaf()) {
      // We need the number of internal nodes with no grandchildren in
      // the hypothetical tree that would be pruned to make 'node' a
      // leaf.  If 'node' is not a leaf right now then we can leave
      // 'number_of_parents_of_leaves' alone.  If it is a leaf, then
      // we need to consider whether the tree obtained by splitting on
      // 'node' would introduce another 'parent_of_leaves' node.  If
      // node->parent() is a parent_of_leaves then the number of
      // parents of leaves remains unchanged (the 'parent_of_leaves'
      // label slides down from the parent to 'node').  Otherwise, the
      // split adds a new 'parent of leaves'.
      const TreeNode *parent = node->parent();
      if (!parent) {
        // 'tree' is currently a singleton, so after implicitly
        // splitting it will have one parent_of_leaves.
        number_of_parents_of_leaves = 1;
      } else {
        if (parent->has_no_grandchildren()) {
          // The parent is a parent of leaves, so adding the split at
          // leaf moves the parent_of_leaves label from parent to
          // leaf, leaving the number of parent_of_leaves nodes
          // unchanged.  Thus we do nothing here.
        } else {
          // If the parent has grandchildren then it is not a parent
          // of leaves, so by splitting on leaf we would add one more
          // parent of leaves.
          ++number_of_parents_of_leaves;
        }
      }
    }
    return log_probability_of_proposing_death_move
        - log_integer(number_of_parents_of_leaves);
  }

  //----------------------------------------------------------------------
  // p(split) = alpha / (1 + d)^beta
  double BartPosteriorSamplerBase::log_probability_of_split(
      int depth) const {
    return log_prior_tree_depth_alpha_
        - prior_tree_depth_beta_
        * log_integer(1 + depth);
  }

  //----------------------------------------------------------------------
  double BartPosteriorSamplerBase::log_probability_of_no_split(
      int depth) const {
    double psplit = prior_tree_depth_alpha_ /
        pow(1 + depth, prior_tree_depth_beta_);
    return log(1 - psplit);
  }

  //----------------------------------------------------------------------
  const GaussianModel * BartPosteriorSamplerBase::node_mean_prior() const {
    return node_mean_prior_.get();
  }

  //----------------------------------------------------------------------
  void BartPosteriorSamplerBase::clear_data_from_trees() {
    for (int i = 0; i < model_->number_of_trees(); ++i) {
      model_->tree(i)->clear_data_and_delete_suf();
    }
  }

}  // namespace BOOM
