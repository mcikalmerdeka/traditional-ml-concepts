"""
Decision tree implementations from scratch.

Includes both Decision Tree Classifier and Regressor.
"""

import numpy as np
from typing import Optional, Literal


class Node:
    """
    Node class for decision tree structure.
    
    Attributes:
        feature: Feature index to split on
        threshold: Threshold value for the split
        left: Left child node
        right: Right child node
        value: Prediction value (class index for classifier leaves, mean
            target for regressor leaves)
        proba: Per-leaf class fractions (classifier leaves only; None for
            regressor and split nodes)
    """
    
    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional['Node'] = None,
        right: Optional['Node'] = None,
        value: Optional[float] = None,
        proba: Optional[np.ndarray] = None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.proba = proba
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf node."""
        return self.value is not None


class DecisionTreeClassifierScratch:
    """
    Decision Tree Classifier implementation from scratch.
    
    Recursively splits on Gini (or entropy) impurity. Leaf nodes store the
    full class distribution, powering both ``predict`` (majority class) and
    ``predict_proba`` (fractions).
    
    Labels are mapped to contiguous indices internally (``classes_`` sorted),
    so non-contiguous or string-coercible labels work like sklearn's trees.
    """
    
    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: Literal['gini', 'entropy'] = 'gini'
    ):
        """
        Initialize Decision Tree Classifier.
        
        Parameters:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required in a leaf node
            criterion: Split quality measure ('gini' or 'entropy')
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.root = None
        self.n_classes_ = None
        self.classes_ = None
    
    def _entropy(self, y: np.ndarray) -> float:
        """Calculate entropy of a node."""
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p + 1e-10) for p in proportions if p > 0])
        return entropy
    
    def _gini(self, y: np.ndarray) -> float:
        """Calculate Gini impurity of a node."""
        proportions = np.bincount(y) / len(y)
        gini = 1 - np.sum(proportions ** 2)
        return gini
    
    def _impurity(self, y: np.ndarray) -> float:
        """Calculate impurity based on criterion."""
        if self.criterion == 'entropy':
            return self._entropy(y)
        else:
            return self._gini(y)
    
    def _information_gain(
        self,
        y: np.ndarray,
        left_y: np.ndarray,
        right_y: np.ndarray
    ) -> float:
        """Calculate information gain from a split."""
        parent_impurity = self._impurity(y)
        n = len(y)
        n_left, n_right = len(left_y), len(right_y)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        # Weighted average of child impurities
        child_impurity = (n_left / n) * self._impurity(left_y) + \
                        (n_right / n) * self._impurity(right_y)
        
        return parent_impurity - child_impurity
    
    def _split(self, X: np.ndarray, threshold: float, feature: int) -> tuple:
        """Split dataset based on feature and threshold."""
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        return left_mask, right_mask
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Find the best split for a node."""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        # Try each feature
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            # Try each unique value as threshold
            for threshold in thresholds:
                left_mask, right_mask = self._split(X, threshold, feature)
                
                # Skip if split doesn't satisfy min_samples_leaf
                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                left_y, right_y = y[left_mask], y[right_mask]
                
                # Calculate information gain
                gain = self._information_gain(y, left_y, right_y)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _leaf(self, y: np.ndarray) -> Node:
        """Create a leaf holding the majority class index and class fractions."""
        counts = np.bincount(y, minlength=self.n_classes_)
        return Node(value=int(np.argmax(counts)), proba=counts / len(y))

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if depth >= self.max_depth or \
           n_samples < self.min_samples_split or \
           n_classes == 1:
            # Create leaf node
            return self._leaf(y)
        
        # Find best split
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        # If no good split found, create leaf
        if best_feature is None or best_gain == 0:
            return self._leaf(y)
        
        # Split data
        left_mask, right_mask = self._split(X, best_threshold, best_feature)
        
        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeClassifierScratch':
        """
        Fit the decision tree classifier.
        
        Parameters:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)
        
        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        # Map arbitrary labels to sorted 0..C-1 indices; leaf logic uses
        # np.bincount, which only accepts non-negative contiguous integers.
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        y_idx = np.searchsorted(self.classes_, y)
        self.root = self._build_tree(X, y_idx)
        return self
    
    def _predict_sample(self, x: np.ndarray, node: Node) -> int:
        """Predict class index for a single sample (mapped back by predict)."""
        if node.is_leaf():
            return int(node.value)
        
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def _leaf_for_sample(self, x: np.ndarray, node: Node) -> Node:
        """Return the leaf node a single sample falls into."""
        if node.is_leaf():
            return node
        
        if x[node.feature] <= node.threshold:
            return self._leaf_for_sample(x, node.left)
        else:
            return self._leaf_for_sample(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.
        
        Parameters:
            X: Features of shape (n_samples, n_features)
        
        Returns:
            Predicted class labels of shape (n_samples,)
        """
        X = np.asarray(X)
        indices = np.array([self._predict_sample(x, self.root) for x in X])
        return self.classes_[indices]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities as the leaf class fractions.
        
        Note: each row is the exact class distribution of the leaf the
        sample falls into, matching scikit-learn's decision tree.
        
        Parameters:
            X: Features of shape (n_samples, n_features)
        
        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        X = np.asarray(X)
        leaves = [self._leaf_for_sample(x, self.root) for x in X]
        return np.array([leaf.proba for leaf in leaves])
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def get_depth(self, node: Optional[Node] = None) -> int:
        """Get the depth of the tree."""
        if node is None:
            node = self.root
        
        if node.is_leaf():
            return 0
        
        left_depth = self.get_depth(node.left) if node.left else 0
        right_depth = self.get_depth(node.right) if node.right else 0
        
        return 1 + max(left_depth, right_depth)
    
    def count_nodes(self, node: Optional[Node] = None) -> int:
        """Count total number of nodes in the tree."""
        if node is None:
            node = self.root
        
        if node.is_leaf():
            return 1
        
        left_count = self.count_nodes(node.left) if node.left else 0
        right_count = self.count_nodes(node.right) if node.right else 0
        
        return 1 + left_count + right_count


class DecisionTreeRegressorScratch:
    """
    Decision Tree Regressor implementation from scratch.
    
    Uses Mean Squared Error (MSE) for splitting.
    """
    
    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1
    ):
        """
        Initialize Decision Tree Regressor.
        
        Parameters:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required in a leaf node
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
    
    def _mse(self, y: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        if len(y) == 0:
            return 0
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)
    
    def _variance_reduction(
        self,
        y: np.ndarray,
        left_y: np.ndarray,
        right_y: np.ndarray
    ) -> float:
        """Calculate variance reduction from a split."""
        parent_mse = self._mse(y)
        n = len(y)
        n_left, n_right = len(left_y), len(right_y)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        # Weighted average of child MSE
        child_mse = (n_left / n) * self._mse(left_y) + \
                   (n_right / n) * self._mse(right_y)
        
        return parent_mse - child_mse
    
    def _split(self, X: np.ndarray, threshold: float, feature: int) -> tuple:
        """Split dataset based on feature and threshold."""
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        return left_mask, right_mask
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Find the best split for a node."""
        best_reduction = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        # Try each feature
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            # Try each unique value as threshold
            for threshold in thresholds:
                left_mask, right_mask = self._split(X, threshold, feature)
                
                # Skip if split doesn't satisfy min_samples_leaf
                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                left_y, right_y = y[left_mask], y[right_mask]
                
                # Calculate variance reduction
                reduction = self._variance_reduction(y, left_y, right_y)
                
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_reduction
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Recursively build the decision tree."""
        n_samples = X.shape[0]
        
        # Stopping criteria
        if depth >= self.max_depth or \
           n_samples < self.min_samples_split or \
           self._mse(y) < 1e-7:  # Nearly pure node
            # Create leaf node with mean value
            leaf_value = np.mean(y)
            return Node(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold, best_reduction = self._best_split(X, y)
        
        # If no good split found, create leaf
        if best_feature is None or best_reduction <= 0:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)
        
        # Split data
        left_mask, right_mask = self._split(X, best_threshold, best_feature)
        
        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeRegressorScratch':
        """
        Fit the decision tree regressor.
        
        Parameters:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,)
        
        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        self.root = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x: np.ndarray, node: Node) -> float:
        """Predict value for a single sample."""
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for samples.
        
        Parameters:
            X: Features of shape (n_samples, n_features)
        
        Returns:
            Predicted values of shape (n_samples,)
        """
        X = np.asarray(X)
        return np.array([self._predict_sample(x, self.root) for x in X])
    
    def apply(self, X: np.ndarray) -> np.ndarray:
        """
        Return the leaf Node object each sample falls into.
        
        Mirrors scikit-learn's ``tree.apply`` concept, but returns the
        scratch ``Node`` objects themselves so stage-wise learners (e.g.
        gradient boosting) can regroup samples per leaf and overwrite the
        leaf values after fitting.
        
        Parameters:
            X: Features of shape (n_samples, n_features)
        
        Returns:
            Array of leaf Node references of shape (n_samples,)
        """
        X = np.asarray(X)
        return np.array(
            [self._predict_sample_leaf(x, self.root) for x in X], dtype=object
        )
    
    def _predict_sample_leaf(self, x: np.ndarray, node: Node) -> Node:
        """Traverse to the leaf for a single sample."""
        if node.is_leaf():
            return node
        if x[node.feature] <= node.threshold:
            return self._predict_sample_leaf(x, node.left)
        return self._predict_sample_leaf(x, node.right)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score."""
        y = np.asarray(y)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_depth(self, node: Optional[Node] = None) -> int:
        """Get the depth of the tree."""
        if node is None:
            node = self.root
        
        if node.is_leaf():
            return 0
        
        left_depth = self.get_depth(node.left) if node.left else 0
        right_depth = self.get_depth(node.right) if node.right else 0
        
        return 1 + max(left_depth, right_depth)
    
    def count_nodes(self, node: Optional[Node] = None) -> int:
        """Count total number of nodes in the tree."""
        if node is None:
            node = self.root
        
        if node.is_leaf():
            return 1
        
        left_count = self.count_nodes(node.left) if node.left else 0
        right_count = self.count_nodes(node.right) if node.right else 0
        
        return 1 + left_count + right_count
