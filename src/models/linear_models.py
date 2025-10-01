"""
Linear models implemented from scratch using NumPy.
Includes: Linear Regression, Ridge, Lasso, Logistic Regression
"""

import numpy as np
from typing import Optional, Union


class LinearRegressionScratch:
    """
    Linear Regression implementation from scratch using NumPy.
    
    Fits model: y = X @ w + b
    Using Normal Equation: w = (X.T @ X)^-1 @ X.T @ y
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    
    Attributes
    ----------
    coef_ : np.ndarray
        Coefficients of the model (slopes)
    intercept_ : float
        Intercept term (bias)
    
    Examples
    --------
    >>> from src.models.linear_models import LinearRegressionScratch
    >>> import numpy as np
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> y = np.array([6, 8, 9, 11])
    >>> model = LinearRegressionScratch()
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)
    >>> print(f"R² Score: {model.score(X, y):.4f}")
    """
    
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionScratch':
        """
        Fit linear model using Normal Equation.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
        y : np.ndarray, shape (n_samples,)
            Target values
        
        Returns
        -------
        self : LinearRegressionScratch
            Fitted estimator
        """
        # Convert to numpy arrays if needed
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Add intercept column if needed
        if self.fit_intercept:
            X_with_intercept = np.c_[np.ones(X.shape[0]), X]
        else:
            X_with_intercept = X
        
        # Normal Equation: w = (X.T @ X)^-1 @ X.T @ y
        # Using pseudo-inverse for numerical stability
        self.coef_ = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        
        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        else:
            self.intercept_ = 0.0
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the linear model.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Samples
        
        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted values
        """
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² (coefficient of determination) score.
        
        Parameters
        ----------
        X : np.ndarray
            Test samples
        y : np.ndarray
            True values
        
        Returns
        -------
        score : float
            R² score (1.0 is perfect prediction)
        """
        y = np.asarray(y)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class RidgeRegressionScratch:
    """
    Ridge Regression (L2 regularization) from scratch.
    
    Adds penalty: alpha * ||w||²
    
    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength; must be positive.
        Larger values specify stronger regularization.
    fit_intercept : bool, default=True
        Whether to fit intercept
    
    Attributes
    ----------
    coef_ : np.ndarray
        Coefficients of the model
    intercept_ : float
        Intercept term
    """
    
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeRegressionScratch':
        """Fit Ridge regression model."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        if self.fit_intercept:
            X_with_intercept = np.c_[np.ones(X.shape[0]), X]
        else:
            X_with_intercept = X
        
        n_features = X_with_intercept.shape[1]
        identity = np.eye(n_features)
        
        # Don't regularize intercept
        if self.fit_intercept:
            identity[0, 0] = 0
        
        # Ridge solution: w = (X.T @ X + alpha * I)^-1 @ X.T @ y
        self.coef_ = np.linalg.pinv(
            X_with_intercept.T @ X_with_intercept + self.alpha * identity
        ) @ X_with_intercept.T @ y
        
        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        else:
            self.intercept_ = 0.0
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using Ridge regression model."""
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score."""
        y = np.asarray(y)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class LassoRegressionScratch:
    """
    Lasso Regression (L1 regularization) from scratch.
    
    Uses coordinate descent for optimization.
    Adds penalty: alpha * ||w||₁
    
    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength
    max_iter : int, default=1000
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for stopping criterion
    fit_intercept : bool, default=True
        Whether to fit intercept
    
    Attributes
    ----------
    coef_ : np.ndarray
        Coefficients of the model
    intercept_ : float
        Intercept term
    """
    
    def __init__(self, alpha: float = 1.0, max_iter: int = 1000, 
                 tol: float = 1e-4, fit_intercept: bool = True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
    
    def _soft_threshold(self, x: float, lambda_: float) -> float:
        """Soft thresholding operator for L1 regularization."""
        if x > lambda_:
            return x - lambda_
        elif x < -lambda_:
            return x + lambda_
        else:
            return 0.0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LassoRegressionScratch':
        """Fit Lasso regression using coordinate descent."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples, n_features = X.shape
        
        # Center data if fitting intercept
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X = X - X_mean
            y = y - y_mean
        else:
            X_mean = np.zeros(n_features)
            y_mean = 0.0
        
        # Initialize coefficients
        self.coef_ = np.zeros(n_features)
        
        # Coordinate descent
        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()
            
            for j in range(n_features):
                # Calculate residual without feature j
                residual = y - (X @ self.coef_ - X[:, j] * self.coef_[j])
                
                # Calculate correlation
                rho = X[:, j] @ residual
                
                # Update coefficient with soft thresholding
                self.coef_[j] = self._soft_threshold(
                    rho / n_samples, 
                    self.alpha
                ) / (np.sum(X[:, j] ** 2) / n_samples)
            
            # Check convergence
            if np.sum(np.abs(self.coef_ - coef_old)) < self.tol:
                break
        
        # Calculate intercept
        if self.fit_intercept:
            self.intercept_ = y_mean - X_mean @ self.coef_
        else:
            self.intercept_ = 0.0
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using Lasso regression model."""
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score."""
        y = np.asarray(y)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class LogisticRegressionScratch:
    """
    Logistic Regression implementation from scratch using NumPy.
    
    Uses gradient descent for optimization.
    Model: p(y=1|x) = sigmoid(w^T x + b)
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    max_iter : int, default=1000
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for stopping criterion
    fit_intercept : bool, default=True
        Whether to fit intercept
    
    Attributes
    ----------
    coef_ : np.ndarray
        Coefficients of the model
    intercept_ : float
        Intercept term
    losses_ : list
        Loss at each iteration (for debugging)
    
    Examples
    --------
    >>> from src.models.linear_models import LogisticRegressionScratch
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 3], [3, 1], [4, 2]])
    >>> y = np.array([0, 0, 1, 1])
    >>> model = LogisticRegressionScratch()
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)
    >>> probabilities = model.predict_proba(X)
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000, 
                 tol: float = 1e-4, fit_intercept: bool = True):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.losses_ = []
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_loss(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss.
        Loss = -mean(y*log(p) + (1-y)*log(1-p))
        """
        # Clip predictions to prevent log(0)
        epsilon = 1e-15
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred_proba) + 
                       (1 - y_true) * np.log(1 - y_pred_proba))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionScratch':
        """
        Fit logistic regression using gradient descent.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
        y : np.ndarray, shape (n_samples,)
            Target values (0 or 1)
        
        Returns
        -------
        self : LogisticRegressionScratch
            Fitted estimator
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0
        
        # Gradient descent
        for iteration in range(self.max_iter):
            # Forward pass
            linear_pred = X @ self.coef_ + self.intercept_
            y_pred_proba = self._sigmoid(linear_pred)
            
            # Compute loss
            loss = self._compute_loss(y, y_pred_proba)
            self.losses_.append(loss)
            
            # Compute gradients
            error = y_pred_proba - y
            grad_coef = (1 / n_samples) * (X.T @ error)
            grad_intercept = (1 / n_samples) * np.sum(error)
            
            # Update parameters
            self.coef_ -= self.learning_rate * grad_coef
            if self.fit_intercept:
                self.intercept_ -= self.learning_rate * grad_intercept
            
            # Check convergence
            if iteration > 0 and abs(self.losses_[-1] - self.losses_[-2]) < self.tol:
                break
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Samples
        
        Returns
        -------
        proba : np.ndarray, shape (n_samples, 2)
            Probabilities for class 0 and class 1
        """
        X = np.asarray(X)
        linear_pred = X @ self.coef_ + self.intercept_
        prob_class_1 = self._sigmoid(linear_pred)
        prob_class_0 = 1 - prob_class_1
        return np.column_stack([prob_class_0, prob_class_1])
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Samples
        threshold : float, default=0.5
            Decision threshold
        
        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Parameters
        ----------
        X : np.ndarray
            Test samples
        y : np.ndarray
            True labels
        
        Returns
        -------
        score : float
            Accuracy score
        """
        y = np.asarray(y)
        y_pred = self.predict(X)
        return np.mean(y == y_pred)
