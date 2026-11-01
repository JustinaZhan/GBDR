"""Granular-ball density ratio (GBDR) for weakly supervised anomaly detection.

Fit with an unlabeled matrix and a separate matrix of labeled anomalies. All
labeled anomalies must be excluded from the unlabeled matrix. Apply the same
preprocessing to both matrices and to future queries before calling this API.
Larger returned log density ratios indicate greater anomaly evidence.

Dependencies: NumPy, SciPy, and scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from sklearn.cluster import KMeans

__all__ = ["GBDR"]

_LAMBDA_GRID = (0.1, 1.0, 10.0)
_GRADIENT_TOLERANCE = 1.0e-6
_BATCH_SIZE = 2048


def _finite_matrix(values, name, n_features=None):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not values.shape[1]:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    if n_features is not None and values.shape[1] != n_features:
        raise ValueError(f"{name} has incompatible feature dimensions")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains nonfinite values")
    return np.ascontiguousarray(values)


def _objective_gradient(theta, phi_u, anomaly_mean, regularization):
    theta = np.asarray(theta, dtype=float)
    logits = phi_u @ theta
    lse = logsumexp(logits)
    probabilities = np.exp(logits - lse)
    value = (lse - math.log(len(phi_u)) - float(anomaly_mean @ theta)
             + 0.5 * regularization * float(theta @ theta))
    gradient = phi_u.T @ probabilities - anomaly_mean + regularization * theta
    return float(value), gradient


@dataclass
class _CoefficientFit:
    theta: np.ndarray
    log_normalizer: float
    objective: float
    gradient_norm: float
    n_iterations: int
    newton_steps: int


def _fit_coefficients(phi_u, phi_a, regularization, max_iter):
    """Solve the same strongly convex objective used for every validation fit."""
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and strictly positive")
    phi_u = _finite_matrix(phi_u, "phi_u")
    phi_a = _finite_matrix(phi_a, "phi_anomaly", phi_u.shape[1])
    anomaly_mean = phi_a.mean(axis=0)
    initial = np.zeros(phi_u.shape[1], dtype=float)
    total_iterations = 0
    newton_budget = min(500, max_iter // 2)
    lbfgs_budget = max(1, max_iter - newton_budget)
    for attempt in range(3):
        result = minimize(
            _objective_gradient, initial,
            args=(phi_u, anomaly_mean, float(regularization)),
            method="L-BFGS-B", jac=True,
            options={"maxiter": lbfgs_budget - total_iterations,
                     "ftol": 1.0e-14 if attempt == 0 else 0.0,
                     "gtol": 1.0e-10, "maxls": 100, "maxcor": 30},
        )
        total_iterations += int(result.nit)
        value, gradient = _objective_gradient(
            result.x, phi_u, anomaly_mean, regularization)
        residual = float(np.max(np.abs(gradient), initial=0.0))
        if (np.isfinite(value) and np.isfinite(result.x).all()
                and residual <= _GRADIENT_TOLERANCE):
            break
        if total_iterations >= lbfgs_budget:
            break
        initial = np.asarray(result.x, dtype=float)
    theta = np.asarray(result.x, dtype=float)
    newton_steps = 0
    # Objective changes can reach machine precision before the gradient is
    # accurate enough. Polish the identical objective using its covariance
    # Hessian, with a reserved budget inside the same total iteration limit.
    while (np.isfinite(value) and np.isfinite(theta).all()
           and residual > _GRADIENT_TOLERANCE and newton_steps < newton_budget
           and total_iterations < max_iter):
        logits = phi_u @ theta
        weights = np.exp(logits - logsumexp(logits))
        weighted_mean = phi_u.T @ weights
        centered = phi_u - weighted_mean[None, :]
        hessian = centered.T @ (weights[:, None] * centered)
        hessian.flat[:: len(theta) + 1] += regularization
        direction = -np.linalg.solve(hessian, gradient)
        slope = float(gradient @ direction)
        if not np.isfinite(direction).all() or slope >= 0:
            break
        accepted = False
        slack = 8 * np.finfo(float).eps * max(1.0, abs(value))
        for backtrack in range(60):
            step = 0.5 ** backtrack
            trial_theta = theta + step * direction
            trial_value, trial_gradient = _objective_gradient(
                trial_theta, phi_u, anomaly_mean, regularization)
            trial_residual = float(np.max(np.abs(trial_gradient), initial=0.0))
            armijo = trial_value <= value + 1.0e-4 * step * slope
            precision_polish = trial_value <= value + slack and trial_residual < residual
            if np.isfinite(trial_value) and (armijo or precision_polish):
                theta, value, gradient, residual = (
                    trial_theta, trial_value, trial_gradient, trial_residual)
                accepted = True
                newton_steps += 1
                total_iterations += 1
                break
        if not accepted:
            break
    if (not np.isfinite(value) or not np.isfinite(theta).all()
            or residual > _GRADIENT_TOLERANCE):
        raise RuntimeError(f"GBDR fit fails the fixed gradient check: {residual:.6g}")
    log_normalizer = float(logsumexp(phi_u @ theta) - math.log(len(phi_u)))
    return _CoefficientFit(theta, log_normalizer, value, residual,
                           total_iterations, newton_steps)


@dataclass
class _Node:
    indices: np.ndarray
    center: np.ndarray
    left: int = -1
    right: int = -1
    split_kind: str = "leaf"
    split_axis: int = -1
    split_value: float = np.nan
    route_left_center: np.ndarray | None = None
    route_right_center: np.ndarray | None = None


class GBDR:
    """Estimate a regularized anomaly-to-unlabeled density ratio.

    Parameters
    ----------
    regularization : float or None, default=None
        A fixed positive coefficient penalty. None selects among 0.1, 1, and
        10 by leave-one-labeled-anomaly-out predictive log ratio. With one
        labeled anomaly the penalty is fixed at 1. Exact ties favor the larger
        penalty. The unlabeled representation remains fixed in every fold.
    random_state : int, default=0
        Seed for deterministic node-specific two-means initializations.
    max_iter : int, default=5000
        Total coefficient-optimization iteration budget per fit, including
        L-BFGS restarts and Newton refinement. The convergence threshold is
        fixed at a gradient infinity norm of 1e-6.

    Fitted attributes
    -----------------
    lambda_, centers_, variances_, mean_, scale_, active_, theta_,
    log_normalizer_, gradient_norm_, n_iter_, newton_steps_, cv_results_
        The selected penalty, leaf centers and squared radial scales,
        unlabeled basis column means and population standard deviations,
        active-column mask, coefficients, empirical log normalization,
        final optimization diagnostics, and parameter-selection criteria.
        Columns with standard deviation at most 1e-12 are exactly zero.
    """

    def __init__(self, regularization=None, random_state=0, max_iter=5000):
        if regularization is not None:
            regularization = float(regularization)
            if not np.isfinite(regularization) or regularization <= 0:
                raise ValueError("regularization must be finite and strictly positive")
        if int(max_iter) != max_iter or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        self.regularization = regularization
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)

    def _threshold_split(self, X, idx):
        local = X[idx]
        axis = int(np.argmax(np.var(local, axis=0)))
        order = np.argsort(local[:, axis], kind="mergesort")
        ordered_values = local[order, axis]
        boundaries = np.flatnonzero(ordered_values[:-1] < ordered_values[1:]) + 1
        h = int(self.min_leaf_size_)
        valid = boundaries[(boundaries >= h) & (boundaries <= len(idx) - h)]
        if not len(valid):
            return None
        cut = int(valid[np.argmin(np.abs(valid - len(idx) / 2.0))])
        split_value = float(ordered_values[cut - 1])
        return (np.sort(idx[order[:cut]]), np.sort(idx[order[cut:]]),
                "threshold", axis, split_value, None, None)

    def _balanced_split(self, X, idx, node_id):
        local = X[idx]
        if len(idx) < 2 * int(self.min_leaf_size_):
            return None
        if np.max(np.ptp(local, axis=0), initial=0.0) <= 1e-12:
            return None
        km = KMeans(n_clusters=2, n_init=10,
                    random_state=self.random_state + 104729 * (node_id + 1))
        labels = km.fit_predict(local)
        if min(np.count_nonzero(labels == 0), np.count_nonzero(labels == 1)) < int(self.min_leaf_size_):
            return self._threshold_split(X, idx)
        c_left = np.asarray(km.cluster_centers_[0], dtype=float)
        c_right = np.asarray(km.cluster_centers_[1], dtype=float)
        differing = np.flatnonzero(np.abs(c_left - c_right) > 1e-12)
        if not differing.size:
            return self._threshold_split(X, idx)
        if c_left[differing[0]] > c_right[differing[0]]:
            c_left, c_right = c_right, c_left
        left_distance = np.sum((local - c_left) ** 2, axis=1)
        right_distance = np.sum((local - c_right) ** 2, axis=1)
        choose_left = left_distance <= right_distance
        left, right = idx[choose_left], idx[~choose_left]
        if min(len(left), len(right)) < int(self.min_leaf_size_):
            return self._threshold_split(X, idx)
        return (np.sort(left), np.sort(right), "centroid", -1, np.nan,
                c_left.copy(), c_right.copy())

    def _build_tree(self, X):
        self.min_leaf_size_ = max(4, int(np.ceil(np.sqrt(len(X)))))
        self._nodes_ = [_Node(np.arange(len(X), dtype=int), X.mean(axis=0))]
        stack = [0]
        while stack:
            node_id = stack.pop()
            node = self._nodes_[node_id]
            split = self._balanced_split(X, node.indices, node_id)
            if split is None:
                continue
            (left_idx, right_idx, node.split_kind, node.split_axis,
             node.split_value, node.route_left_center,
             node.route_right_center) = split
            child_ids = []
            for child_idx in (left_idx, right_idx):
                child_ids.append(len(self._nodes_))
                self._nodes_.append(_Node(child_idx, X[child_idx].mean(axis=0)))
            node.left, node.right = child_ids
            stack.extend(reversed(child_ids))
        self._leaf_ids_ = [i for i, node in enumerate(self._nodes_) if node.left < 0]

    def _raw_transform(self, X):
        out = np.empty((len(X), len(self.centers_)), dtype=float)
        for start in range(0, len(X), _BATCH_SIZE):
            stop = min(len(X), start + _BATCH_SIZE)
            distance_squared = cdist(X[start:stop], self.centers_, metric="sqeuclidean")
            out[start:stop] = np.exp(-distance_squared / (2.0 * self.variances_[None, :]))
        return out

    def _standardize(self, raw_phi):
        out = np.zeros_like(raw_phi)
        out[:, self.active_] = ((raw_phi[:, self.active_] - self.mean_[self.active_])
                               / self.scale_[self.active_])
        return out

    def _transform(self, X):
        X = _finite_matrix(X, "X_query", self.n_features_in_)
        return self._standardize(self._raw_transform(X))

    def fit(self, X_unlabeled, X_anomaly):
        """Fit on disjoint, finite, consistently preprocessed sample matrices."""
        X_u = _finite_matrix(X_unlabeled, "X_unlabeled")
        X_a = _finite_matrix(X_anomaly, "X_anomaly", X_u.shape[1])
        # A failed refit must not score a new representation with old weights.
        for name in ("theta_", "log_normalizer_", "lambda_", "gradient_norm_",
                     "n_iter_", "newton_steps_"):
            self.__dict__.pop(name, None)
        self.n_features_in_ = X_u.shape[1]
        global_center = X_u.mean(axis=0)
        global_scatter = float(np.mean(np.sum((X_u - global_center) ** 2, axis=1)))
        variance_floor = 1.0e-12 * max(global_scatter, 1.0)
        self._build_tree(X_u)
        members = [self._nodes_[i].indices for i in self._leaf_ids_]
        self.centers_ = np.asarray([self._nodes_[i].center for i in self._leaf_ids_], dtype=float)
        raw_variances = [float(np.mean(np.sum((X_u[idx] - center) ** 2, axis=1)))
                         for idx, center in zip(members, self.centers_)]
        self.variances_ = np.maximum(np.asarray(raw_variances, dtype=float), variance_floor)
        raw_phi = self._raw_transform(X_u)
        self.mean_ = raw_phi.mean(axis=0)
        self.scale_ = raw_phi.std(axis=0, ddof=0)
        self.active_ = self.scale_ > 1.0e-12
        self._phi_unlabeled_ = self._standardize(raw_phi)
        phi_a = self._transform(X_a)
        self.cv_results_ = []
        # Never overwrite the construction parameter: repeated fit calls must
        # repeat selection when regularization=None, just as fresh instances do.
        if self.regularization is None:
            if len(X_a) == 1:
                selected_regularization = 1.0
            else:
                for regularization in _LAMBDA_GRID:
                    values = []
                    for heldout in range(len(X_a)):
                        training = np.arange(len(X_a)) != heldout
                        fitted = _fit_coefficients(
                            self._phi_unlabeled_, phi_a[training],
                            regularization, self.max_iter)
                        values.append(float(phi_a[heldout] @ fitted.theta - fitted.log_normalizer))
                    self.cv_results_.append({
                        "regularization": regularization,
                        "mean_heldout_log_ratio": float(np.mean(values)),
                    })
                winner = max(self.cv_results_, key=lambda row: (
                    row["mean_heldout_log_ratio"], row["regularization"]))
                selected_regularization = float(winner["regularization"])
        else:
            selected_regularization = self.regularization
        fitted = _fit_coefficients(self._phi_unlabeled_, phi_a, selected_regularization, self.max_iter)
        self.lambda_ = selected_regularization
        self.theta_ = fitted.theta
        self.log_normalizer_ = fitted.log_normalizer
        self.gradient_norm_ = fitted.gradient_norm
        self.n_iter_ = fitted.n_iterations
        self.newton_steps_ = fitted.newton_steps
        return self

    def decision_function(self, X):
        """Return log density ratios; larger values indicate greater anomaly evidence."""
        if not hasattr(self, "theta_"):
            raise RuntimeError("GBDR must be fitted before scoring")
        return self._transform(X) @ self.theta_ - self.log_normalizer_

    def score_samples(self, X):
        """Alias of decision_function, with the same larger-is-more-anomalous sign."""
        return self.decision_function(X)
