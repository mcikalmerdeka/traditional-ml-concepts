"""
K-Nearest Neighbors implementations from scratch.

Includes both KNN Classifier and Regressor.
"""

import numpy as np
from typing import Optional, Literal
from collections import Counter


class KNNClassifierScratch:
    """
    K-Nearest Neighbors Classifier implementation from scratch.
    
    Uses distance metrics to find k nearest neighbors and predicts
    based on majority voting.
    """
    
    def __init__(
        self,
        n_neighbors: int = 5,
        metric: Literal['euclidean', 'manhattan', 'minkowski'] = 'euclidean',
        p: int = 2,
        weights: Literal['uniform', 'distance'] = 'uniform'
    ):
        """
        Initialize KNN Classifier.
        
        Parameters:
            n_neighbors: Number of neighbors to use
            metric: Distance metric ('euclidean', 'manhattan', 'minkowski')
            p: Power parameter for Minkowski distance
            weights: Weight function ('uniform' or 'distance')
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.weights = weights
        self.X_train = None
        self.y_train = None
    
    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _manhattan_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Manhattan distance between two points."""
        return np.sum(np.abs(x1 - x2))
    
    def _minkowski_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Minkowski distance between two points."""
        return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)
    
    def _calculate_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate distance based on selected metric."""
        if self.metric == 'euclidean':
            return self._euclidean_distance(x1, x2)
        elif self.metric == 'manhattan':
            return self._manhattan_distance(x1, x2)
        elif self.metric == 'minkowski':
            return self._minkowski_distance(x1, x2)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNClassifierScratch':
        """
        Fit the KNN classifier (just stores training data).
        
        Parameters:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)
        
        Returns:
            self
        """
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self
    
    def _predict_single(self, x: np.ndarray) -> int:
        """Predict class for a single sample."""
        # Calculate distances to all training samples
        distances = [self._calculate_distance(x, x_train) 
                    for x_train in self.X_train]
        
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.n_neighbors]
        
        # Get labels of k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]
        
        if self.weights == 'uniform':
            # Majority voting
            most_common = Counter(k_nearest_labels).most_common(1)
            return most_common[0][0]
        else:  # distance weighted
            # Weight by inverse distance (avoid division by zero)
            k_distances = np.array([distances[i] for i in k_indices])
            weights = 1 / (k_distances + 1e-10)
            
            # Weighted voting
            unique_labels = np.unique(k_nearest_labels)
            weighted_votes = {}
            for label in unique_labels:
                mask = k_nearest_labels == label
                weighted_votes[label] = np.sum(weights[mask])
            
            return max(weighted_votes, key=weighted_votes.get)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.
        
        Parameters:
            X: Features of shape (n_samples, n_features)
        
        Returns:
            Predicted class labels of shape (n_samples,)
        """
        X = np.asarray(X)
        return np.array([self._predict_single(x) for x in X])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples.
        
        Parameters:
            X: Features of shape (n_samples, n_features)
        
        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        X = np.asarray(X)
        n_classes = len(np.unique(self.y_train))
        probabilities = []
        
        for x in X:
            # Calculate distances to all training samples
            distances = [self._calculate_distance(x, x_train) 
                        for x_train in self.X_train]
            
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_labels = self.y_train[k_indices]
            
            if self.weights == 'uniform':
                # Count votes for each class
                proba = np.zeros(n_classes)
                for label in k_nearest_labels:
                    proba[label] += 1
                proba /= self.n_neighbors
            else:  # distance weighted
                k_distances = np.array([distances[i] for i in k_indices])
                weights = 1 / (k_distances + 1e-10)
                
                proba = np.zeros(n_classes)
                for label, weight in zip(k_nearest_labels, weights):
                    proba[label] += weight
                proba /= np.sum(proba)
            
            probabilities.append(proba)
        
        return np.array(probabilities)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class KNNRegressorScratch:
    """
    K-Nearest Neighbors Regressor implementation from scratch.
    
    Uses distance metrics to find k nearest neighbors and predicts
    based on averaging (weighted or uniform).
    """
    
    def __init__(
        self,
        n_neighbors: int = 5,
        metric: Literal['euclidean', 'manhattan', 'minkowski'] = 'euclidean',
        p: int = 2,
        weights: Literal['uniform', 'distance'] = 'uniform'
    ):
        """
        Initialize KNN Regressor.
        
        Parameters:
            n_neighbors: Number of neighbors to use
            metric: Distance metric ('euclidean', 'manhattan', 'minkowski')
            p: Power parameter for Minkowski distance
            weights: Weight function ('uniform' or 'distance')
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.weights = weights
        self.X_train = None
        self.y_train = None
    
    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _manhattan_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Manhattan distance between two points."""
        return np.sum(np.abs(x1 - x2))
    
    def _minkowski_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Minkowski distance between two points."""
        return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)
    
    def _calculate_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate distance based on selected metric."""
        if self.metric == 'euclidean':
            return self._euclidean_distance(x1, x2)
        elif self.metric == 'manhattan':
            return self._manhattan_distance(x1, x2)
        elif self.metric == 'minkowski':
            return self._minkowski_distance(x1, x2)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNRegressorScratch':
        """
        Fit the KNN regressor (just stores training data).
        
        Parameters:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,)
        
        Returns:
            self
        """
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self
    
    def _predict_single(self, x: np.ndarray) -> float:
        """Predict value for a single sample."""
        # Calculate distances to all training samples
        distances = [self._calculate_distance(x, x_train) 
                    for x_train in self.X_train]
        
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.n_neighbors]
        
        # Get values of k nearest neighbors
        k_nearest_values = self.y_train[k_indices]
        
        if self.weights == 'uniform':
            # Simple average
            return np.mean(k_nearest_values)
        else:  # distance weighted
            # Weight by inverse distance
            k_distances = np.array([distances[i] for i in k_indices])
            weights = 1 / (k_distances + 1e-10)
            
            # Weighted average
            return np.sum(weights * k_nearest_values) / np.sum(weights)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for samples.
        
        Parameters:
            X: Features of shape (n_samples, n_features)
        
        Returns:
            Predicted values of shape (n_samples,)
        """
        X = np.asarray(X)
        return np.array([self._predict_single(x) for x in X])
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score."""
        y = np.asarray(y)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

