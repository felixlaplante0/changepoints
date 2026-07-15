"""Duality-based changepoint detection."""

from numbers import Real
from typing import Any, Self

import numpy as np
import numpy.typing as npt
from numba import njit
from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import Interval, validate_params
from sklearn.utils.validation import check_is_fitted

from .costs._costs import BaseCost, GaussianMeanCost
from .utils._validation import validate_X


class DUST(BaseEstimator):
    """Detects changepoints with the one-dimensional DUST algorithm."""

    @validate_params(
        {
            "cost": [type],
            "penalty": [Interval(Real, 0, None, closed="left")],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        cost: type[BaseCost] = GaussianMeanCost,
        penalty: float = 1,
        **kwargs: Any,
    ):
        """Initializes the DUST estimator.

        Args:
            cost (type[BaseCost], optional): The one-parameter cost function. Defaults
                to GaussianMeanCost.
            penalty (float, optional): The penalty per segment. Defaults to 1.
            kwargs (Any): Additional arguments passed to ``cost``.
        """
        self.cost = cost
        self.penalty = penalty
        self.kwargs = kwargs

    def _get_fit(self, X: np.ndarray):
        """Builds the compiled DUST recursion for ``X``."""
        cost_object = self.cost(X, **self.kwargs)
        if "dust" not in vars(cost_object):
            raise ValueError("DUST requires a one-parameter cost function")
        cost = cost_object.__call__
        dust = cost_object.dust

        def _fit_impl(penalty: float):
            n = X.shape[0]
            buffer = np.empty(n)
            candidates = np.empty(n, dtype=np.int64)
            keep = np.empty(n, dtype=np.bool_)
            costs = np.zeros(n)
            previous = np.zeros(n, dtype=np.int64)
            candidates[0] = 0
            size = 1

            for t in range(1, n):
                for j in range(size):
                    s = candidates[j]
                    buffer[j] = cost(costs[s], s, t)

                best = 0
                for j in range(1, size):
                    if buffer[j] < buffer[best]:
                        best = j
                costs[t] = buffer[best] + penalty
                previous[t] = candidates[best]

                keep[0] = buffer[0] <= costs[t]
                for j in range(1, size):
                    keep[j] = not dust(costs, candidates[j - 1], candidates[j], t)

                write = 0
                for j in range(size):
                    if keep[j]:
                        candidates[write] = candidates[j]
                        write += 1
                candidates[write] = t
                size = write + 1

            changepoints = np.empty(n, dtype=np.int64)
            size = 0
            t = n - 1
            while t > 0:
                t = previous[t]
                if t > 0:
                    changepoints[size] = t
                    size += 1
            return changepoints[:size][::-1], costs

        return njit(fastmath=True)(_fit_impl)  # type: ignore

    @validate_params(
        {"X": ["array-like"], "y": [None]},
        prefer_skip_nested_validation=True,
    )
    def fit(self, X: npt.ArrayLike, y: None = None) -> Self:  # noqa: ARG002
        """Fits DUST to a univariate signal.

        Args:
            X (npt.ArrayLike): A two-dimensional signal with one column.
            y (None, optional): Ignored. Defaults to None.

        Raises:
            ValueError: If ``X`` is not univariate or the cost is not one-parameter.

        Returns:
            Self: The fitted estimator.
        """
        X = validate_X(X)  # type: ignore
        if X.shape[1] != 1:
            raise ValueError("DUST requires X with exactly one column")

        key = (
            hash(X.tobytes()),
            self.cost,
            repr(sorted(self.kwargs.items())),
        )
        if getattr(self, "_kernel_key", None) != key:
            self._kernel = self._get_fit(X)
            self._kernel_key = key
        self.chgpts, self.min_costs = self._kernel(self.penalty)

        return self

    def predict(self, X: Any = None) -> np.ndarray:  # noqa: ARG002
        """Returns the fitted changepoints.

        Args:
            X (None, optional): Ignored. Defaults to None.

        Returns:
            np.ndarray: The changepoint indices.
        """
        check_is_fitted(self, "chgpts")
        return self.chgpts

    @validate_params(
        {"X": ["array-like"], "y": [None]},
        prefer_skip_nested_validation=True,
    )
    def fit_predict(self, X: npt.ArrayLike, y: None = None) -> np.ndarray:  # noqa: ARG002
        """Fits DUST and returns the changepoints.

        Args:
            X (npt.ArrayLike): A two-dimensional signal with one column.
            y (None, optional): Ignored. Defaults to None.

        Returns:
            np.ndarray: The changepoint indices.
        """
        return self.fit(X).predict()
