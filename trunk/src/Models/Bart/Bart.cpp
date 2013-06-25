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

#include <algorithm>
#include <iterator>

#include <Models/Bart/Bart.hpp>
#include <Models/Bart/ResidualRegressionData.hpp>
#include <cpputil/math_utils.hpp>
#include <cpputil/report_error.hpp>
#include <distributions.hpp>
#include <stats/moments.hpp>

namespace BOOM {
  namespace Bart {
    namespace {
      inline void remove_node_and_descendents_from_set(
          TreeNode *node,
          std::set<TreeNode *> &set_of_nodes) {
        if (!node) {
          return;
        }
        set_of_nodes.erase(node);
        if (!node->is_leaf()) {
          remove_node_and_descendents_from_set(
              node->left_child(),
              set_of_nodes);
          remove_node_and_descendents_from_set(
              node->right_child(),
              set_of_nodes);
        }
      }
    }

    //----------------------------------------------------------------------
    VariableSummary::VariableSummary(int variable_number)
        : variable_number_(variable_number)
    {}

    //----------------------------------------------------------------------
    VariableSummary::VariableSummary(
        const SerializedVariableSummary &serialized)
    {
      deserialize(serialized);
    }

    //----------------------------------------------------------------------
    void VariableSummary::observe_value(double value) {
      observed_values_.push_back(value);
    }

    //----------------------------------------------------------------------
    bool VariableSummary::random_cutpoint(
        RNG &rng,
        const TreeNode *node,
        double *cutpoint) const {
      check_finalized("random_cutpoint");
      return impl_->random_cutpoint(rng, node, cutpoint);
    }

    //----------------------------------------------------------------------
    void VariableSummary::finalize(
        int discrete_distribution_cutoff,
        ContinuousCutpointStrategy strategy) {
      observed_values_.sort();
      Vector::iterator end =
          std::unique(observed_values_.begin(), observed_values_.end());

      int number_of_unique_values =
          std::distance(observed_values_.begin(), end);

      if (number_of_unique_values > discrete_distribution_cutoff) {
        // Assume the variable being summarized is continuous.
        switch (strategy) {
          case UNIFORM_CONTINUOUS:
            if (number_of_unique_values < observed_values_.size()) {
              observed_values_.erase(end, observed_values_.end());
            }
            impl_.reset(new ContinuousVariableSummary(variable_number_,
                                                      observed_values_));
            break;
          case UNIFORM_DISCRETE:
            if (number_of_unique_values < observed_values_.size()) {
              observed_values_.erase(end, observed_values_.end());
            }
            impl_.reset(new DiscreteVariableSummary(variable_number_,
                                                    observed_values_));
            break;
          default:
            report_error("Unknown enum value passed to "
                         "VariableSummary::finalize");
        }
      } else {
        // Assume the variable being summarized is discrete.
        if (number_of_unique_values < observed_values_.size()) {
          observed_values_.erase(end, observed_values_.end());
        }
        impl_.reset(new DiscreteVariableSummary(variable_number_,
                                                observed_values_));
      }
      observed_values_.clear();
    }

    //----------------------------------------------------------------------
    SerializedVariableSummary VariableSummary::serialize() const {
      if (!impl_) {
        SerializedVariableSummary ans;
        ans.finalized = false;
        ans.data = observed_values_;
        ans.variable_number = variable_number_;
        return ans;
      } else {
        return impl_->serialize();
      }
    }

    //----------------------------------------------------------------------
    void VariableSummary::deserialize(
        const SerializedVariableSummary &serialized) {
      variable_number_ = serialized.variable_number;
      // If the variable summary had not been finalized, then just
      // grab the observed values and return.
      if (!serialized.finalized) {
        observed_values_ = serialized.data;
        return;
      }

      // If the variable summary had been finalized then proceed from
      // here.
      if (serialized.is_continuous) {
        switch(serialized.strategy) {
          case UNIFORM_CONTINUOUS:
            impl_.reset(new ContinuousVariableSummary(
                serialized.variable_number,
                serialized.data));
            break;
          case UNIFORM_DISCRETE:
            impl_.reset(new DiscreteVariableSummary(
                serialized.variable_number,
                serialized.data));
            break;
          default:
            report_error("Unknown enum value passed to VariableSummary::set.");
        }
      } else {
        impl_.reset(new DiscreteVariableSummary(
            serialized.variable_number,
            serialized.data));
      }
    }

    //----------------------------------------------------------------------
    void VariableSummary::check_finalized(const char *msg) const {
      if (!impl_) {
        ostringstream err;
        err << "A VariableSummary must be finalized before calling "
            << msg << endl;
        report_error(err.str());
      }
    }

    //======================================================================
    VariableSummaryImpl::VariableSummaryImpl(int variable_index)
        : variable_index_(variable_index)
    {}

    //======================================================================
    DiscreteVariableSummary::DiscreteVariableSummary(int variable_index,
                                                     const Vector &values)
        : VariableSummaryImpl(variable_index),
          cutpoint_values_(values)
    {
      cutpoint_values_.sort();
      Vector::iterator garbage_begin =
          std::unique(cutpoint_values_.begin(), cutpoint_values_.end());
      cutpoint_values_.erase(garbage_begin, cutpoint_values_.end());
    }

    // The algorithm used here is random sampling with bisection.  We
    // keep track of two integers: lo and hi, indicating the smallest
    // and largest potential values.  These start just inside at the
    // global lower and upper bounds for the variable (e.g. at
    // position 0 and one before back()).  If a random chosen index is
    // out of bounds (based on the observed lower and upper bounds)
    // then the bounds are moved accordingly, and we try again with
    // different bounds.
    //
    // It might be faster to use the STL algorithms lower_bound and
    // upper_bound to find the bounds and then do random number
    // generation.
    bool DiscreteVariableSummary::random_cutpoint(
        RNG &rng,
        const TreeNode *node,
        double *cutpoint) const {
      if (cutpoint_values_.size() < 2) {
        return false;
      }
      double original_lower_bound = cutpoint_values_[0] - 1;
      double lower_bound = original_lower_bound;
      double original_upper_bound = cutpoint_values_.back();
      double upper_bound = original_upper_bound;

      node->get_cutpoint_range(variable_index(),
                               &lower_bound,
                               &upper_bound);

      // If nothing has been done to alter the default lower bound,
      // then the first cutpoint is still available.
      int lo = 0;
      if (lower_bound > original_lower_bound) {
        // If the lower bound has been altered by node's ancestors
        // then we need to find the position of the lower bound
        // observed "in the wild".  The next largest cutpoint is the
        // first acceptable value.
        //
        // The standard library lower_bound algorithm is a fast way to
        // search through the set of cutpoint_values_, which are
        // stored as an ordered set.
        lo = std::distance(cutpoint_values_.begin(),
                           std::lower_bound(cutpoint_values_.begin(),
                                            cutpoint_values_.end(),
                                            lower_bound));
        ++lo;
      }

      // If nothing has been done to alter the default upper bound
      // then the next to last cutpoint is the largest acceptable
      // value, because we can't split on the largest cutpoint with a
      // <= splitting rule, because no data would go to the right.
      int hi = cutpoint_values_.size() - 2;
      if (upper_bound < original_upper_bound) {
        // If the upper_bound has been altered, we need to find the
        // position of the upper_bound seen "in the wild".  The next
        // smallest cutpoint is the right hand endpoint of the set of
        // acceptable values.
        hi = std::distance(cutpoint_values_.begin(),
                           std::lower_bound(cutpoint_values_.begin(),
                                            cutpoint_values_.end(),
                                            upper_bound));
        --hi;
      }
      if (hi < lo) {
        return false;
      } else if (hi == lo) {
        *cutpoint = cutpoint_values_[hi];
        return true;
      } else {
        int pos = random_int_mt(rng, lo, hi);
        *cutpoint = cutpoint_values_[pos];
        return true;
      }
    }

    //----------------------------------------------------------------------
    void DiscreteVariableSummary::set_cutpoint_values(const Vector &values) {
      cutpoint_values_ = values;
    }

    //----------------------------------------------------------------------
    SerializedVariableSummary DiscreteVariableSummary::serialize() const {
      SerializedVariableSummary ans;
      ans.finalized = true;
      ans.variable_number = variable_index();
      ans.is_continuous = false;
      ans.strategy = UNIFORM_DISCRETE;
      ans.data = cutpoint_values_;
      return ans;
    }

    //======================================================================
    ContinuousVariableSummary::ContinuousVariableSummary(
        int variable_index,
        const Vector &values)
        : VariableSummaryImpl(variable_index),
          lo_(values[0]),
          hi_(values.back())
    {}

    //----------------------------------------------------------------------
    bool ContinuousVariableSummary::random_cutpoint(
        RNG &rng,
        const TreeNode *node,
        double *cutpoint) const {
      double lower_bound = lo_;
      double upper_bound = hi_;
      node->get_cutpoint_range(variable_index(),
                               &lower_bound,
                               &upper_bound);
      if (lower_bound >= upper_bound) {
        return false;
      }
      *cutpoint = runif_mt(rng, lower_bound, upper_bound);
      return true;
    }

    //----------------------------------------------------------------------
    SerializedVariableSummary ContinuousVariableSummary::serialize() const {
      SerializedVariableSummary ans;
      ans.finalized = true;
      ans.variable_number = variable_index();
      ans.is_continuous = true;
      ans.strategy = UNIFORM_CONTINUOUS;
      ans.data.resize(2);
      ans.data[0] = lo_;
      ans.data[1] = hi_;
      return ans;
    }

    //======================================================================

    TreeNode::TreeNode(double mean_value, TreeNode *parent)
        : parent_(parent),
          left_child_(NULL),
          right_child_(NULL),
          depth_(parent_ ? 1 + parent_->depth() : 0),
          mean_(mean_value),
          which_variable_(-1),             // needs to be set
          cutpoint_(BOOM::infinity())      // needs to be set
    {}

    //----------------------------------------------------------------------
    TreeNode::~TreeNode() {
      prune_descendants();
    }

    //----------------------------------------------------------------------
    TreeNode * TreeNode::recursive_clone(TreeNode *parent) {
      TreeNode *copy = new TreeNode(mean_, parent);
      if (left_child_) {
        copy->left_child_ = left_child_->recursive_clone(this);
      }
      if (right_child_) {
        copy->right_child_ = right_child_->recursive_clone(this);
      }
      copy->which_variable_ = this->which_variable_;
      copy->cutpoint_ = this->cutpoint_;
      return copy;
    }

    //----------------------------------------------------------------------
    bool TreeNode::operator==(const TreeNode &rhs) const {
      if (is_leaf()) {
        return rhs.is_leaf() && mean() == rhs.mean();
      } else {
        return which_variable_ == rhs.which_variable_
            && !rhs.is_leaf()
            && *left_child_ == *(rhs.left_child_)
            && *right_child_ == *(rhs.right_child_);
      }
    }

    //----------------------------------------------------------------------
    bool TreeNode::operator!=(const TreeNode &rhs) const {
      return !(*this == rhs);
    }

    //----------------------------------------------------------------------
    double TreeNode::predict(const Vector &x) const {
      return predict(ConstVectorView(x));
    }

    //----------------------------------------------------------------------
    double TreeNode::predict(const VectorView &x) const {
      return predict(ConstVectorView(x));
    }

    //----------------------------------------------------------------------
    double TreeNode::predict(const ConstVectorView &x) const {
      if (is_leaf()) {
        return mean_;
      } else {
        if (x[which_variable_] <= cutpoint_) {
          return left_child_->predict(x);
        } else {
          return right_child_->predict(x);
        }
      }
    }

    //----------------------------------------------------------------------
    void TreeNode::grow(double left_mean_value, double right_mean_value) {
      if (!is_leaf()) {
        ostringstream err;
        err << "TreeNode::grow() called on an interior node.  "
            << "It should only be called on leaves.";
        report_error(err.str());
      }
      left_child_ = new TreeNode(left_mean_value, this);
      right_child_ = new TreeNode(right_mean_value, this);
      if (!!suf_) {
        left_child_->populate_sufficient_statistics(suf_->create());
        right_child_->populate_sufficient_statistics(suf_->create());
      }
      for (int i = 0; i < data_.size(); ++i) {
        ResidualRegressionData *dp(data_[i]);
        if (dp->x()[which_variable_] <= cutpoint_) {
          left_child_->populate_data(dp, false);
        } else {
          right_child_->populate_data(dp, false);
        }
      }
    }

    //----------------------------------------------------------------------
    int TreeNode::prune_descendants() {
      int number_pruned = 0;
      if (left_child_) {
        number_pruned += left_child_->prune_descendants();
        delete left_child_;
        left_child_ = NULL;
        ++number_pruned;
      }
      if (right_child_) {
        number_pruned += right_child_->prune_descendants();
        delete right_child_;
        right_child_ = NULL;
        ++number_pruned;
      }
      return number_pruned;
    }

    //----------------------------------------------------------------------
    bool TreeNode::is_leaf() const {
      // Since the tree is a binary tree, it is enough to check that
      // there is no left child, because if there is no left child there
      // can be no right child either.
      return left_child_ == NULL;
    }

    //----------------------------------------------------------------------
    bool TreeNode::has_no_grandchildren() const {
      return is_leaf() ||
          (left_child_->is_leaf()
           && right_child_->is_leaf());
    }

    //----------------------------------------------------------------------
    int TreeNode::depth() const {
      return depth_;
    }

    //----------------------------------------------------------------------
    bool TreeNode::is_left_child() const {
      if (parent_ == NULL) return false;
      return this == parent_->left_child_;
    }

    //----------------------------------------------------------------------
    bool TreeNode::is_right_child() const {
      if (parent_ == NULL) return false;
      return this == parent_->right_child_;
    }

    //----------------------------------------------------------------------
    TreeNode * TreeNode::parent() {
      return parent_;
    }

    //----------------------------------------------------------------------
    const TreeNode * TreeNode::parent() const {
      return parent_;
    }

    //----------------------------------------------------------------------
    TreeNode * TreeNode::left_child() {
      return left_child_;
    }

    //----------------------------------------------------------------------
    TreeNode * TreeNode::right_child() {
      return right_child_;
    }

    //----------------------------------------------------------------------
    void TreeNode::get_cutpoint_range(
        int variable_index,
        double *lower_cutpoint_bound,
        double *upper_cutpoint_bound) const {
      if (*lower_cutpoint_bound >= *upper_cutpoint_bound) {
        // The node is logically empty.  Note the equality.  If the
        // lower bound is equal to the upper bound then there are no
        // splits remaining.
        return;
      }

      if (parent_ == NULL) {
        // Quit looking if you've reached the root of the tree.
        return;
      }

      if (parent_->which_variable_ == variable_index) {
        if (is_left_child()) {
          // All data flowing to this node will be less than the
          // parent's cutpoint, which sets a new upper bound.
          *upper_cutpoint_bound = std::min<double>(
              *upper_cutpoint_bound,
              parent_->cutpoint_);

        } else if (is_right_child()) {
          // All data flowing to this node will be greater than the
          // parent's cutpoint, which sets a new lower bound.
          *lower_cutpoint_bound = std::max<double>(
              *lower_cutpoint_bound,
              parent_->cutpoint_);
        }
      }
      parent_->get_cutpoint_range(variable_index,
                                  lower_cutpoint_bound,
                                  upper_cutpoint_bound);
    }

    //----------------------------------------------------------------------
    void TreeNode::set_variable_and_cutpoint(int variable_index,
                                             double cutpoint) {
      which_variable_ = variable_index;
      cutpoint_ = cutpoint;
    }

    //----------------------------------------------------------------------
    void TreeNode::set_mean(double value) {
      mean_ = value;
    }

    //----------------------------------------------------------------------
    double TreeNode::mean() const {
      return mean_;
    }

    //----------------------------------------------------------------------
    int TreeNode::variable_index() const {
      return which_variable_;
    }

    //----------------------------------------------------------------------
    double TreeNode::cutpoint() const {
      return cutpoint_;
    }

    //----------------------------------------------------------------------
    void TreeNode::clear_data_and_delete_suf(bool recursive) {
      data_.clear();
      if (!!suf_) {
        suf_.reset();
      }
      if (recursive) {
        if (left_child_) {
          left_child_->clear_data_and_delete_suf(recursive);
        }
        if (right_child_) {
          right_child_->clear_data_and_delete_suf(recursive);
        }
      }
    }

    //----------------------------------------------------------------------
    void TreeNode::populate_sufficient_statistics(
        SufficientStatisticsBase *suf,
        bool recursive) {
      suf_.reset(suf);
      if (recursive && !is_leaf()) {
        left_child_->populate_sufficient_statistics(suf->clone(), recursive);
        right_child_->populate_sufficient_statistics(suf->clone(), recursive);
      }
    }

    //----------------------------------------------------------------------
    void TreeNode::populate_data(ResidualRegressionData *dp, bool recursive) {
      data_.push_back(dp);
      if (recursive && !is_leaf()) {
        const Vector &x(dp->x());
        if (x[which_variable_] <= cutpoint_) {
          left_child_->populate_data(dp, recursive);
        } else {
          right_child_->populate_data(dp, recursive);
        }
      }
    }

    //----------------------------------------------------------------------
    const SufficientStatisticsBase * TreeNode::compute_suf() {
      if (!!suf_) {
        suf_->clear();
      }
      for (int i = 0; i < data_.size(); ++i) {
        suf_->update(*(data_[i]));
      }
      return suf_.get();
    }

    //----------------------------------------------------------------------
    const std::vector<ResidualRegressionData *> & TreeNode::data() const {
      return data_;
    }

    //----------------------------------------------------------------------
    void TreeNode::remove_mean_effect() {
      for (int i = 0; i < data_.size(); ++i) {
        data_[i]->add_to_residual(mean_);
      }
    }

    //----------------------------------------------------------------------
    void TreeNode::replace_mean_effect() {
      for (int i = 0; i < data_.size(); ++i) {
        data_[i]->subtract_from_residual(mean_);
      }
    }

    //----------------------------------------------------------------------
    ostream & TreeNode::print(ostream &out) const {
      for (int i = 0; i < depth_; ++i) {
        out << ".";
      }
      if (is_leaf()) {
        out << " " << mean_ << endl;
      } else {
        out << "v" << which_variable_
            << "(" << cutpoint_ << ")" << endl;
        left_child_->print(out);
        right_child_->print(out);
      }
      return out;
    }

    //----------------------------------------------------------------------
    int TreeNode::fill_tree_matrix_row(int parent_id,
                                       int my_id,
                                       Matrix *tree_matrix) const {
      VectorView row(tree_matrix->row(my_id));
      bool leaf = this->is_leaf();
      row[0] = parent_id;
      row[1] = mean_;
      row[2] = leaf ? -1 : which_variable_;
      row[3] = leaf ? infinity() : cutpoint_;
      int next_id = my_id + 1;
      if (!leaf) {
        next_id = left_child_->fill_tree_matrix_row(
            my_id, next_id, tree_matrix);
        next_id = right_child_->fill_tree_matrix_row(
            my_id, next_id, tree_matrix);
      }
      return next_id;
    }

    //======================================================================

    Tree::Tree(double mean_value)
        : root_(new TreeNode(mean_value)),
          number_of_nodes_(1)
    {
      leaves_.insert(root_.get());
    }

    //----------------------------------------------------------------------
    Tree::Tree(const Matrix &tree_matrix)
        : number_of_nodes_(nrow(tree_matrix))
    {
      std::vector<TreeNode *> nodes(number_of_nodes_);
      for (int id = 0; id < number_of_nodes_; ++id) {
        const ConstVectorView node_info(tree_matrix.row(id));
        int parent_id = lround(node_info[0]);
        double mean = node_info[1];
        int variable_index = lround(node_info[2]);
        double cutpoint = node_info[3];

        // This scheme relies on the fact that each node's id is
        // greater than its parent's id.
        TreeNode *parent = (parent_id >= 0) ? nodes[parent_id] : NULL;
        TreeNode *node = new TreeNode(mean, parent);
        node->set_variable_and_cutpoint(variable_index, cutpoint);
        if (parent_id >= 0) {
          if (id == parent_id + 1) {
            parent->left_child_ = node;
          } else {
            parent->right_child_ = node;
          }
        }
        nodes[id] = node;
      }
      root_.reset(nodes[0]);
      register_special_nodes(root_.get());
    }

    //----------------------------------------------------------------------
    Tree::Tree(const Tree &rhs)
        : root_(rhs.root_->recursive_clone(NULL)),
          number_of_nodes_(rhs.number_of_nodes_)
    {
      register_special_nodes(root_.get());
    }

    //----------------------------------------------------------------------
    Tree & Tree::operator=(const Tree &rhs) {
      if (&rhs != this) {
        root_.reset(rhs.root_->recursive_clone(NULL));
        number_of_nodes_ = rhs.number_of_nodes_;
        register_special_nodes(root_.get());
      }
      return *this;
    }

    //----------------------------------------------------------------------
    void Tree::swap(Tree &rhs) {
      root_.swap(rhs.root_);
      std::swap<int>(number_of_nodes_, rhs.number_of_nodes_);
      leaves_.swap(rhs.leaves_);
      parents_of_leaves_.swap(rhs.parents_of_leaves_);
    }

    //----------------------------------------------------------------------
    Tree::~Tree() {
      root_.reset();
    }

    //----------------------------------------------------------------------
    bool Tree::operator==(const Tree &rhs) const {
      return *(root_) == *(rhs.root_);
    }

    //----------------------------------------------------------------------
    bool Tree::operator!=(const Tree &rhs) const {
      return !(*this == rhs);
    }

    //----------------------------------------------------------------------
    double Tree::predict(const Vector &x) const {
      return root_->predict(x);
    }

    //----------------------------------------------------------------------
    double Tree::predict(const VectorView &x) const {
      return root_->predict(x);
    }

    //----------------------------------------------------------------------
    double Tree::predict(const ConstVectorView &x) const {
      return root_->predict(x);
    }

    //----------------------------------------------------------------------
    int Tree::number_of_nodes() const {
      return number_of_nodes_;
    }

    //----------------------------------------------------------------------
    int Tree::number_of_leaves() const {
      return leaves_.size();
    }

    //----------------------------------------------------------------------
    Tree::NodeSetIterator Tree::leaf_begin() {
      return leaves_.begin();
    }

    //----------------------------------------------------------------------
    Tree::NodeSetIterator Tree::leaf_end() {
      return leaves_.end();
    }

    //----------------------------------------------------------------------
    TreeNode * Tree::random_leaf(RNG &rng) {
      int n = leaves_.size();
      int which = random_int_mt(rng, 0, n-1);
      Tree::NodeSetIterator iterator = leaf_begin();
      std::advance(iterator, which);
      TreeNode *leaf = *iterator;
      if (!leaf->is_leaf()) {
        ostringstream err;
        err << "Tree::random_leaf() found an answer that is not a leaf:" << endl
            << "The returned value is: "<< endl
            << *leaf << endl
            << "The tree is " << endl
            << *this;
        report_error(err.str());
      }
      return leaf;
    }

    //----------------------------------------------------------------------
    int Tree::number_of_parents_of_leaves() const {
      return parents_of_leaves_.size();
    }

    //----------------------------------------------------------------------
    Tree::NodeSetIterator Tree::parents_of_leaves_begin() {
      return parents_of_leaves_.begin();
    }

    //----------------------------------------------------------------------
    Tree::NodeSetIterator Tree::parents_of_leaves_end() {
      return parents_of_leaves_.end();
    }

    //----------------------------------------------------------------------
    TreeNode * Tree::random_parent_of_leaves(RNG &rng) {
      if (root_->is_leaf()) {
        return NULL;
      }
      int n = parents_of_leaves_.size();
      int which = random_int_mt(rng, 0, n - 1);
      Tree::NodeSetIterator iterator = parents_of_leaves_begin();
      std::advance(iterator, which);
      return *iterator;
    }

    //----------------------------------------------------------------------
    // Grow the tree at the given leaf, and adjust the sets of special
    // nodes.
    void Tree::grow(TreeNode *leaf, double left_mean, double right_mean) {
      if (!leaf->is_leaf()) {
        ostringstream err;
        err << "The node " << endl << *leaf << " is not a leaf in this tree: "
            << endl
            << *this;
        report_error(err.str());
      }

      bool found = leaves_.erase(leaf);
      if (!found) {
        ostringstream err;
        err << "Tree::grow called on a leaf that was not "
            "managed by the tree."
            << endl << *this;
        report_error(err.str());
      }

      if (leaf->parent()) {
        // The parent might or might not already have grandchildren
        // (because there may be grandchildren on the other branch).
        // It is about to get some, so if it is in the set of
        // parents_of_leaves_ we need to take it
        // out.
        if (leaf->parent()->has_no_grandchildren()) {
          parents_of_leaves_.erase(leaf->parent());
        }
      }

      parents_of_leaves_.insert(leaf);
      leaf->grow(left_mean, right_mean);
      leaves_.insert(leaf->left_child());
      leaves_.insert(leaf->right_child());
      number_of_nodes_ += 2;
    }

    //----------------------------------------------------------------------
    // Removes any descendents from node.  Descendents are also
    // removed from leaves_ or parents_of_leaves_.
    // Node will be added to the set of leaves.
    void Tree::prune_descendants(TreeNode *node) {
      remove_node_and_descendents_from_set(node->left_child(), leaves_);
      remove_node_and_descendents_from_set(node->right_child(), leaves_);
      remove_node_and_descendents_from_set(node, parents_of_leaves_);
      if (node->parent() && node->parent()->has_no_grandchildren()) {
        parents_of_leaves_.insert(node->parent());
      }
      leaves_.insert(node);
      number_of_nodes_ -= node->prune_descendants();
    }

    //----------------------------------------------------------------------
    void Tree::populate_sufficient_statistics(SufficientStatisticsBase *suf) {
      root_->populate_sufficient_statistics(suf, true);
    }

    //----------------------------------------------------------------------
    void Tree::populate_data(ResidualRegressionData *data) {
      root_->populate_data(data, true);
    }

    //----------------------------------------------------------------------
    void Tree::clear_data_and_delete_suf() {
      root_->clear_data_and_delete_suf(true);
    }

    //----------------------------------------------------------------------
    void Tree::remove_mean_effect() {
      for (NodeSetIterator it = leaves_.begin(); it != leaves_.end(); ++it) {
        (*it)->remove_mean_effect();
      }
    }

    //----------------------------------------------------------------------
    void Tree::replace_mean_effect() {
      for (NodeSetIterator it = leaves_.begin(); it != leaves_.end(); ++it) {
        (*it)->replace_mean_effect();
      }
    }

    //----------------------------------------------------------------------
    ostream & Tree::print(ostream &out) const {
      return root_->print(out);
    }

    //----------------------------------------------------------------------
    Matrix Tree::to_matrix() const {
      Matrix ans(number_of_nodes(), 4);
      root_->fill_tree_matrix_row(-1, 0, &ans);
      return ans;
    }

    //----------------------------------------------------------------------
    void Tree::from_matrix(const ConstSubMatrix &tree_matrix) {
      number_of_nodes_ = tree_matrix.nrow();
      leaves_.clear();
      parents_of_leaves_.clear();
      std::vector<TreeNode *> nodes(number_of_nodes_);
      for (int id = 0; id < number_of_nodes_; ++id) {
        const ConstVectorView node_info(tree_matrix.row(id));
        int parent_id = lround(node_info[0]);
        double mean = node_info[1];
        int variable_index = lround(node_info[2]);
        double cutpoint = node_info[3];

        // This scheme relies on the fact that each node's id is
        // greater than its parent's id.
        TreeNode *parent = (parent_id >= 0) ? nodes[parent_id] : NULL;
        TreeNode *node = new TreeNode(mean, parent);
        node->set_variable_and_cutpoint(variable_index, cutpoint);
        if (parent_id >= 0) {
          if (id == parent_id + 1) {
            parent->left_child_ = node;
          } else {
            parent->right_child_ = node;
          }
        }
        nodes[id] = node;
      }
      root_.reset(nodes[0]);
      register_special_nodes(root_.get());
    }

    //----------------------------------------------------------------------
    void Tree::register_special_nodes(TreeNode *node) {
      if (node->is_leaf()) {
        leaves_.insert(node);
      } else {
        if (node->has_no_grandchildren()) {
          parents_of_leaves_.insert(node);
        }
        register_special_nodes(node->left_child());
        register_special_nodes(node->right_child());
      }
    }

  } // namespace Bart

  //======================================================================
  BartModelBase::BartModelBase(int number_of_trees, double mean)
  {
    create_trees(number_of_trees, mean);
  }

  //----------------------------------------------------------------------
  BartModelBase::BartModelBase(const BartModelBase &rhs)
      : Model(rhs),
        variable_summaries_(rhs.variable_summaries_),
        trees_(rhs.trees_)
  {
    for (int i = 0; i < trees_.size(); ++i) {
      trees_[i].reset(new Bart::Tree(*(rhs.trees_[i])));
    }
  }

  //----------------------------------------------------------------------
  double BartModelBase::predict(const Vector &x) const {
    return predict(ConstVectorView(x));
  }

  //----------------------------------------------------------------------
  double BartModelBase::predict(const VectorView &x) const {
    return predict(ConstVectorView(x));
  }

  //----------------------------------------------------------------------
  double BartModelBase::predict(const ConstVectorView &x) const {
    double ans = 0;
    for (int i = 0; i < trees_.size(); ++i) {
      ans += trees_[i]->predict(x);
    }
    return ans;
  }

  //----------------------------------------------------------------------
  int BartModelBase::number_of_variables() const {
    return variable_summaries_.size();
  }

  //----------------------------------------------------------------------
  int BartModelBase::number_of_trees() const {
    return trees_.size();
  }

  //----------------------------------------------------------------------
  void BartModelBase::set_number_of_trees(int new_number_of_trees) {
    int trees_needed = new_number_of_trees - number_of_trees();
    if (trees_needed == 0) return;
    else if (trees_needed > 0) {
      add_trees(trees_needed, 0.0);
    } else if (trees_needed < 0) {
      remove_trees(abs(trees_needed));
    } else {
      report_error("Unknown value for new_number_of_trees.");
    }
  }

  //----------------------------------------------------------------------
  void BartModelBase::rebuild_tree(int i, const ConstSubMatrix &matrix) {
    trees_[i]->from_matrix(matrix);
  }

  //----------------------------------------------------------------------
  void BartModelBase::finalize_data(
        int discrete_distribution_cutoff,
        Bart::ContinuousCutpointStrategy strategy) {
    for (int i = 0; i < number_of_variables(); ++i) {
      variable_summaries_[i].finalize(discrete_distribution_cutoff,
                                      strategy);
    }
  }

  //----------------------------------------------------------------------
  const Bart::VariableSummary &
  BartModelBase::variable_summary(int which_variable) const {
    return variable_summaries_[which_variable];
  }

  //----------------------------------------------------------------------
  void BartModelBase::set_variable_summaries(
      const std::vector<Bart::SerializedVariableSummary> &serialized) {
    variable_summaries_.clear();
    variable_summaries_.reserve(serialized.size());
    for (int i = 0; i < serialized.size(); ++i) {
      variable_summaries_.push_back(Bart::VariableSummary(serialized[i]));
    }
  }

  //----------------------------------------------------------------------
  Bart::Tree * BartModelBase::tree(int which_tree) {
    return trees_[which_tree].get();
  }

  //----------------------------------------------------------------------
  const Bart::Tree * BartModelBase::tree(int which_tree) const {
    return trees_[which_tree].get();
  }

  //----------------------------------------------------------------------
  void BartModelBase::observe_data(const Vector &x) {
    ConstVectorView view(x);
    observe_data(view);
  }

  //----------------------------------------------------------------------
  void BartModelBase::observe_data(const ConstVectorView &x) {
    int xdim = x.size();
    check_variable_dimension(xdim);
    for (int i = 0; i < xdim; ++i) {
      variable_summaries_[i].observe_value(x[i]);
    }
  }

  //----------------------------------------------------------------------
  void BartModelBase::check_variable_dimension(int dim) {
    if (variable_summaries_.empty()) {
      variable_summaries_.reserve(dim);
      for (int i = 0; i < dim; ++i) {
        variable_summaries_.push_back(Bart::VariableSummary(i));
      }
    } else {
      if (variable_summaries_.size() != dim) {
        report_error("Wrong sized variable summaries.");
      }
    }
  }

  //----------------------------------------------------------------------
  void BartModelBase::create_trees(int number_of_trees, double mean) {
    trees_.clear();
    add_trees(number_of_trees, mean / number_of_trees);
  }

  //----------------------------------------------------------------------
  void BartModelBase::add_trees(int number_of_additional_trees,
                                    double mean) {
    trees_.reserve(trees_.size() + number_of_additional_trees);
    for (int i = 0; i < number_of_additional_trees; ++i) {
      boost::shared_ptr<Bart::Tree> tree(new Bart::Tree(mean));
      trees_.push_back(tree);
    }
  }

  //----------------------------------------------------------------------
  void BartModelBase::remove_trees(int number_of_trees_to_remove) {
    if (number_of_trees_to_remove >= trees_.size()) {
      trees_.clear();
      return;
    }

    for (int i = 0; i < number_of_trees_to_remove; ++i) {
      trees_.pop_back();
    }
  }

}  // namespace BOOM
