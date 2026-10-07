"""
Multi-layer perceptron classifier implemented from scratch using NumPy.

Includes:
- MLPClassifierScratch: densely connected feed-forward classifier (the
  "basics" neural network; convolutions, recurrent cells, and dropout
  are out of scope for this module).

Core idea: a stack of affine maps with pointwise nonlinearities can
approximate complex class boundaries. Training is end-to-end gradient
descent: backpropagation applies the chain rule layer by layer to turn
the loss into per-weight gradients, and the ADAM optimizer keeps
per-parameter first/second moment estimates (bias-corrected) so every
weight gets an adaptive, roughly scale-free step size.

Simplifications vs. scikit-learn (documented on purpose):
- Full-batch gradient descent: every iteration computes the gradient on
  ALL training rows; sklearn iterates over shuffled minibatches instead.
- Glorot-uniform init drawn layer by layer from
  ``np.random.RandomState(random_state)``. Same init family as sklearn,
  but the exact draws differ, so scratch and sklearn runs are never
  philosophically comparable parameter-by-parameter.
- Early stopping watches the TRAINING loss and keeps the FINAL
  parameters when the loss fails to improve by more than ``tol`` for
  ``n_iter_no_change`` consecutive iterations. sklearn's
  ``early_stopping=True`` instead splits off a validation set and
  reverts to the best-validation parameters (out of scope here).
- Only the adam-style update: no 'sgd' or 'lbfgs' solver, no
  ``partial_fit``/``warm_start``/``batch_size``/``class_weight``.
"""

from typing import List, Optional, Tuple, Union

import numpy as np

_ACTIVATIONS = ("relu", "tanh", "logistic")

_BETA1 = 0.9  # ADAM first-moment decay
_BETA2 = 0.999  # ADAM second-moment decay
_EPSILON = 1e-8  # ADAM denominator guard against division by zero

HiddenLayerSizes = Union[int, Tuple[int, ...], List[int]]


def _activate(name: str, Z: np.ndarray) -> np.ndarray:
    """Apply the configured nonlinearity to a pre-activation batch."""
    if name == "relu":
        return np.maximum(0.0, Z)
    if name == "tanh":
        return np.tanh(Z)
    # 'logistic' = sigmoid
    return 1.0 / (1.0 + np.exp(-Z))


def _activate_backward(name: str, A: np.ndarray) -> np.ndarray:
    """
    Derivative of the activation wrt its pre-activation, expressed
    through the stored activation OUTPUT A so pre-activation values
    never need to be cached during the forward pass.
    """
    if name == "relu":
        return (A > 0).astype(float)
    if name == "tanh":
        return 1.0 - A**2
    # 'logistic': sigmoid'(z) = a * (1 - a)
    return A * (1.0 - A)


class MLPClassifierScratch:
    """
    Feed-forward neural network (dense MLP) classifier from scratch.

    Architecture: input -> [dense hidden layer + activation] for each
    entry of ``hidden_layer_sizes`` -> dense output layer with one unit
    per class and a softmax. Trained with full-batch gradient descent
    on cross-entropy plus L2, using the ADAM optimizer.

    Parameters:
        hidden_layer_sizes: Units per hidden layer, as an int (meaning
            one layer of that size) or a tuple with one positive int
            per hidden layer
        activation: Nonlinearity between hidden layers: 'relu'
            (max(0, x)), 'tanh', or 'logistic' (1 / (1 + exp(-x)))
        alpha: L2 penalty strength on weight matrices; intercepts are
            never penalized (sklearn's convention)
        learning_rate_init: Base step size for the ADAM optimizer
        max_iter: Maximum full-batch gradient-descent iterations
        tol: Minimum improvement over the best loss so far that counts
            as progress for early stopping
        n_iter_no_change: Consecutive non-improving iterations that end
            training
        random_state: Seed for the weight initialization

    Attributes after fit:
        classes_: Sorted unique target labels, shape (C,)
        coefs_: Weight array per layer; ``coefs_[l]`` has shape
            (fan_in, fan_out) and maps activation ``l`` to layer ``l+1``
        intercepts_: Bias array per layer; ``intercepts_[l]`` has shape
            (fan_out,)
        loss_: Final loss logged in ``loss_curve_``
        loss_curve_: Loss at every run iteration (list of float), from
            before update ``1`` to before the last applied update
        n_iter_: Number of iterations actually run
    """

    def __init__(
        self,
        hidden_layer_sizes: HiddenLayerSizes = (100,),
        activation: str = "relu",
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter: int = 200,
        tol: float = 1e-4,
        n_iter_no_change: int = 10,
        random_state: Optional[int] = None,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifierScratch":
        """
        Fit the network with full-batch ADAM gradient descent.

        Parameters:
            X: Training features of shape (n_samples, n_features)
            y: Class labels of shape (n_samples,); arbitrary label
                values are allowed and are mapped to sorted classes_

        Returns:
            self
        """
        self._validate_hyperparams()
        hidden = self._resolve_hidden_sizes()

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Map arbitrary labels to 0..C-1 (same recipe as the forest):
        # softmax columns then correspond to sorted classes_.
        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y)
        n_samples, n_features = X.shape

        # Glorot-uniform init: limit = sqrt(6 / (fan_in + fan_out)).
        # One shared RNG draws weight matrices in layer order, so a
        # fixed random_state fully pins the whole training trajectory.
        layer_dims = [n_features, *hidden, len(self.classes_)]
        rng = np.random.RandomState(self.random_state)
        self.coefs_ = []
        self.intercepts_ = []
        for fan_in, fan_out in zip(layer_dims, layer_dims[1:]):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.coefs_.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)))
            self.intercepts_.append(np.zeros(fan_out))

        self._reset_adam()
        self.loss_curve_ = []
        best_loss = np.inf
        no_improvement = 0

        for iteration in range(1, self.max_iter + 1):
            loss, grads_w, grads_b = self._loss_and_gradient(X, y_idx)
            self.loss_curve_.append(loss)

            # sklearn's plateau semantics on the training loss: progress
            # means dropping below the best loss seen minus tol; anything
            # better opens a fresh window of n_iter_no_change.
            if loss < best_loss - self.tol:
                no_improvement = 0
            else:
                no_improvement += 1
            best_loss = min(best_loss, loss)

            self._apply_adam(iteration, grads_w, grads_b)

            if no_improvement >= self.n_iter_no_change:
                break

        self.n_iter_ = len(self.loss_curve_)
        self.loss_ = self.loss_curve_[-1]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Class probabilities via the softmax output layer.

        Parameters:
            X: Features of shape (n_samples, n_features)

        Returns:
            Probabilities of shape (n_samples, C); column j corresponds
            to ``classes_[j]``
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        _, logits = self._forward(X)
        return self._softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels as the argmax of the probabilities.

        Parameters:
            X: Features of shape (n_samples, n_features)

        Returns:
            Predicted labels from ``classes_`` of shape (n_samples,)
        """
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy on the given data."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def _check_fitted(self) -> None:
        """Fail with a readable message instead of an obscure shape error."""
        if not self.coefs_:
            raise RuntimeError(
                "This MLPClassifierScratch instance is not fitted yet; "
                "call fit(X, y) before predicting."
            )

    def _validate_hyperparams(self) -> None:
        """Validate the scalar hyperparameters at fit time (sklearn-style)."""
        if self.activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {_ACTIVATIONS}; got {self.activation!r}"
            )
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0; got {self.alpha}")
        self._require_positive_int(self.max_iter, "max_iter")
        if self.tol < 0:
            raise ValueError(f"tol must be >= 0; got {self.tol}")
        self._require_positive_int(self.n_iter_no_change, "n_iter_no_change")
        if self.learning_rate_init <= 0:
            raise ValueError(
                f"learning_rate_init must be > 0; got {self.learning_rate_init}"
            )

    @staticmethod
    def _require_positive_int(value, name: str) -> None:
        """Integers >= 1 only; bools are ints in Python and excluded."""
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a positive integer; got {value!r}")
        if value < 1:
            raise ValueError(f"{name} must be a positive integer; got {value!r}")

    def _resolve_hidden_sizes(self) -> Tuple[int, ...]:
        """
        Normalize ``hidden_layer_sizes`` to a tuple of positive ints.

        An int means a single layer of that size; a tuple/list keeps its
        entries (each must be a positive int).
        """
        sizes = self.hidden_layer_sizes
        if isinstance(sizes, bool) or not isinstance(sizes, (int, np.integer)):
            if not isinstance(sizes, (tuple, list)):
                raise ValueError(
                    "hidden_layer_sizes must be an int or a tuple of positive ints; "
                    f"got {sizes!r}"
                )
            sizes = tuple(sizes)
        else:
            sizes = (int(sizes),)

        for size in sizes:
            if (
                isinstance(size, bool)
                or not isinstance(size, (int, np.integer))
                or size < 1
            ):
                raise ValueError(
                    "hidden_layer_sizes must be an int or a tuple of positive ints; "
                    f"got {self.hidden_layer_sizes!r}"
                )
        return tuple(int(size) for size in sizes)

    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Run the network forward.

        Returns the activation batch per layer (``activations[l]`` is
        the input consumed by weight layer ``l``) plus the raw output
        logits; the softmax belongs to the loss/probability stage.
        """
        activations = [X]
        for W, b in zip(self.coefs_[:-1], self.intercepts_[:-1]):
            activations.append(_activate(self.activation, activations[-1] @ W + b))
        logits = activations[-1] @ self.coefs_[-1] + self.intercepts_[-1]
        return activations, logits

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """
        Row-wise softmax with logits max-shifted: exp(z - max) divides
        the same partition as exp(z) but cannot overflow in float64.
        """
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def _loss_and_gradient(
        self, X: np.ndarray, y_idx: np.ndarray
    ) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:
        """
        Loss and analytic gradient at the current parameters.

        L = cross-entropy(mean over rows, gathered in log-space so an
        underflowing probability never produces log(0))
          + alpha * sum of squared weights over ``coefs_``.

        Returns (loss, grad wrt each weight matrix, grad wrt each bias).
        The backprop chain rule yields dL/dlogits = (softmax - onehot)/n
        for the mean cross-entropy; the L2 term contributes 2*alpha*W to
        weight gradients only (the derivative of alpha*||W||^2) and is
        deliberately absent from bias gradients.
        """
        activations, logits = self._forward(X)
        n = len(X)

        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        partition = exp.sum(axis=1, keepdims=True)
        proba = exp / partition
        # log(softmax) = shifted - log(partition): exact, no clipping.
        log_ce = shifted[np.arange(n), y_idx] - np.log(partition)[:, 0]
        loss = -float(np.mean(log_ce))
        loss += self.alpha * sum(float((W**2).sum()) for W in self.coefs_)

        dZ = proba.copy()
        dZ[np.arange(n), y_idx] -= 1.0
        dZ /= n

        grads_w: List[np.ndarray] = [None] * len(self.coefs_)
        grads_b: List[np.ndarray] = [None] * len(self.intercepts_)
        for layer in reversed(range(len(self.coefs_))):
            grads_w[layer] = activations[layer].T @ dZ
            grads_w[layer] += 2.0 * self.alpha * self.coefs_[layer]
            grads_b[layer] = dZ.sum(axis=0)
            if layer > 0:
                dZ = (dZ @ self.coefs_[layer].T) * _activate_backward(
                    self.activation, activations[layer]
                )

        return loss, grads_w, grads_b

    def _reset_adam(self) -> None:
        """Zero the per-parameter moment buffers for a fresh fit."""
        self._adam_m_w = [np.zeros_like(W) for W in self.coefs_]
        self._adam_v_w = [np.zeros_like(W) for W in self.coefs_]
        self._adam_m_b = [np.zeros_like(b) for b in self.intercepts_]
        self._adam_v_b = [np.zeros_like(b) for b in self.intercepts_]

    def _apply_adam(
        self,
        t: int,
        grads_w: List[np.ndarray],
        grads_b: List[np.ndarray],
    ) -> None:
        """Update every parameter with the bias-corrected ADAM rule."""
        self._adam_step(self.coefs_, grads_w, self._adam_m_w, self._adam_v_w, t)
        self._adam_step(self.intercepts_, grads_b, self._adam_m_b, self._adam_v_b, t)

    def _adam_step(
        self,
        params: List[np.ndarray],
        grads: List[np.ndarray],
        moments_m: List[np.ndarray],
        moments_v: List[np.ndarray],
        t: int,
    ) -> None:
        """
        One ADAM step for a parameter list.

        Bias correction divides by 1 - beta**t with t starting at 1:
        the raw moments start at zero and are biased toward zero early
        on, and the correction makes the first step already behave like
        an lr-size step instead of a tiny one.
        """
        for i in range(len(params)):
            moments_m[i] = _BETA1 * moments_m[i] + (1.0 - _BETA1) * grads[i]
            moments_v[i] = _BETA2 * moments_v[i] + (1.0 - _BETA2) * grads[i] ** 2
            m_hat = moments_m[i] / (1.0 - _BETA1**t)
            v_hat = moments_v[i] / (1.0 - _BETA2**t)
            step = self.learning_rate_init * m_hat / (np.sqrt(v_hat) + _EPSILON)
            params[i] = params[i] - step
