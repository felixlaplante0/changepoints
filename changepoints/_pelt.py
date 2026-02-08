from collections.abc import Callable
from numbers import Real
from typing import Any, Self

import numpy as np
import numpy.typing as npt
from numba import config, njit, prange, set_num_threads
from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import Integral, Interval, validate_params
from sklearn.utils.validation import check_is_fitted

from .costs._costs import BaseCost, GaussianMeanCost
from .utils._cache import np_cache
from .utils._validation import validate_X


class PELT(BaseEstimator):
    """A fast implementation of the PELT.

    Attributes:
        cost (type[BaseCost]): The cost to use.
        beta (float): The penalty.
        num_threads (int): The number of threads to use.
        kwargs (Any): Key word arguments used when needed by the base cost.
    """

    cost: type[BaseCost]
    beta: float
    num_threads: int
    kwargs: Any

    @validate_params(
        {
            "cost": [type],
            "beta": [Interval(Real, 0, None, closed="left")],
            "num_threads": [Interval(Integral, 1, None, closed="left"), None],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        cost: type[BaseCost] = GaussianMeanCost,
        beta: float = 1,
        *,
        num_threads: int | None = None,
        **kwargs: Any,
    ):
        """A wrapper for the PELT algorithm.

        Args:
            cost (type[BaseCost], optional): The cost function to use. Defaults to
                GaussianMeanCost.
            beta (float, optional): The penalty parameter. Defaults to 1.0.
            num_threads (int | None, optional): The number of threads to use. If left to
                None, this is computed as the number of available processors divided by
                2. Defaults to None.
            kwargs (Any): Additional kwargs used when needed by the base cost.

        """
        self.cost = cost
        self.beta = beta
        self.num_threads = (
            max(1, config.NUMBA_NUM_THREADS // 2)
            if num_threads is None
            else num_threads
        )
        self.kwargs = kwargs

        set_num_threads(self.num_threads)

    @np_cache
    def _get_write(
        self, X: np.ndarray
    ) -> Callable[[np.ndarray, np.ndarray, int, int], None]:
        """Gets the functions to write in buffer using cache not to repeat compilation.

        Args:
            X (np.ndarray): The data.

        Returns:
            Callable[[np.ndarray, np.ndarray, int, int], None]: The write function.
        """
        cost = self.cost(X, **self.kwargs).__call__

        @njit(fastmath=True)  # type: ignore
        def _write_serial(
            buffer: np.ndarray, min_costs: np.ndarray, start: int, t: int
        ):
            for i in range(start, t):
                buffer[i] = cost(min_costs[i], i, t)

        @njit(fastmath=True, parallel=True)  # type: ignore
        def _write_parallel(
            buffer: np.ndarray, min_costs: np.ndarray, start: int, t: int
        ):
            for i in prange(start, t):
                buffer[i] = cost(min_costs[i], i, t)

        return _write_serial if self.num_threads == 1 else _write_parallel

    @validate_params(
        {
            "X": ["array-like"],
            "y": [None],
        },
        prefer_skip_nested_validation=True,
    )
    def fit(self, X: npt.ArrayLike, y: None = None) -> Self:  # noqa: ARG002
        """Fit the PELT algorithm.

        Args:
            X (npt.ArrayLike): The data.
            y (None, optional): Ignored. Defaults to None.

        Returns:
            Self: The fitted estimator.
        """
        X = validate_X(X)  # type: ignore

        # Initialize variables
        n = X.shape[0]
        start = 0
        buffer = np.empty(n)
        min_costs = np.zeros(n)
        chgpts = np.empty(n, dtype=int)
        write = self._get_write(X)

        for t in range(1, n):
            write(buffer, min_costs, start, t)

            idx = start + np.argmin(buffer[start:t])
            min_costs[t] = (m := buffer[idx] + self.beta)
            chgpts[t] = idx

            while buffer[start] > m:
                start += 1

        final_chpts: list[int] = []
        t = n - 1
        while t > 0:
            final_chpts.append(t := int(chgpts[t]))

        self.chgpts = np.array(final_chpts[:-1][::-1])
        self.min_costs = min_costs

        return self

    def predict(self, X: Any = None) -> np.ndarray:  # noqa: ARG002
        """Predict the change points.

        Args:
            X (None, optional): Ignored. Defaults to None.

        Returns:
            np.ndarray: The change points.
        """
        check_is_fitted(self, "chgpts")
        return self.chgpts

    @validate_params(
        {
            "X": ["array-like"],
            "y": [None],
        },
        prefer_skip_nested_validation=True,
    )
    def fit_predict(self, X: npt.ArrayLike, y: None = None) -> np.ndarray:  # noqa: ARG002
        """Fit and predict the change points.

        Args:
            X (npt.ArrayLike): The data.
            y (None, optional): Ignored. Defaults to None.

        Returns:
            np.ndarray: The change points.
        """
        return self.fit(X).predict()
