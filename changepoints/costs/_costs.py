from typing import Protocol

import numpy as np
from numba import njit


class BaseCost(Protocol):
    """Base class for cost functions."""

    def __init__(self, X: np.ndarray):
        """Initialize the cost of a given segment.

        Args:
            X (np.ndarray): The data.
        """
        ...

    def __call__(self, min_cost: float, idx: int, t: int) -> float:
        """Compute the cost of the given idx.

        Args:
            min_cost (float): The recursively computed minimum cost.
            idx (int): The idx of the potential changepoint.
            t (int): The current time.
        """
        ...


class GaussianMeanCost(BaseCost):
    def __init__(self, X: np.ndarray):
        """Initialize the cost of a given segment for Gaussian data.

        Args:
            X (np.ndarray): The data.
        """
        d = X.shape[1]
        cumsum = np.cumsum(X, axis=0)
        cumsum2 = np.cumsum(X * X, axis=0)

        @njit(fastmath=True)  # type: ignore
        def _cost(min_cost: float, idx: int, t: int) -> float:
            s = 0.0
            n = t - idx

            for j in range(d):
                diff = cumsum[t, j] - cumsum[idx, j]
                diff2 = cumsum2[t, j] - cumsum2[idx, j]
                s += diff2 - (diff * diff) / n

            return s + min_cost

        self.__call__ = _cost


class GaussianVarianceCost(BaseCost):
    def __init__(self, X: np.ndarray):
        """Initialize the cost for variance change detection.

        Args:
            X (np.ndarray): The data.
        """
        d = X.shape[1]
        cumsum2 = np.cumsum(X * X, axis=0)

        @njit(fastmath=True)  # type: ignore
        def _cost(min_cost: float, idx: int, t: int) -> float:
            s = 0.0
            n = t - idx

            for j in range(d):
                diff2 = cumsum2[t, j] - cumsum2[idx, j]
                s += n * np.log(diff2 / n + 1e-8) + n

            return s + min_cost

        self.__call__ = _cost


class GaussianMeanVarianceCost(BaseCost):
    def __init__(self, X: np.ndarray):
        """Initialize the cost for detecting changes in both mean and variance.

        Args:
            X (np.ndarray): The data.
        """
        d = X.shape[1]
        cumsum = np.cumsum(X, axis=0)
        cumsum2 = np.cumsum(X * X, axis=0)

        @njit(fastmath=True)  # type: ignore
        def _cost(min_cost: float, idx: int, t: int) -> float:
            s = 0.0
            n = t - idx

            for j in range(d):
                diff = cumsum[t, j] - cumsum[idx, j]
                diff2 = cumsum2[t, j] - cumsum2[idx, j]
                c = diff2 - (diff * diff) / n
                s += n * np.log(c / n + 1e-8)

            return s + min_cost

        self.__call__ = _cost


class PoissonCost(BaseCost):
    def __init__(self, X: np.ndarray):
        """Initialize the cost of a given segment for Poisson data.

        Args:
            X (np.ndarray): The data (counts).
        """
        d = X.shape[1]
        cumsum = np.cumsum(X, axis=0)

        @njit(fastmath=True)  # type: ignore
        def _cost(min_cost: float, idx: int, t: int) -> float:
            s = 0.0
            n = t - idx

            for j in range(d):
                diff = cumsum[t, j] - cumsum[idx, j]
                s += diff - diff * np.log(diff / n + 1e-8)

            return s + min_cost

        self.__call__ = _cost


class GeometricCost(BaseCost):
    def __init__(self, X: np.ndarray):
        """Initialize the cost of a given segment for Geometric data.

        Args:
            X (np.ndarray): The data (number of trials until the first success).
                X must be integers >= 1.
        """
        d = X.shape[1]
        cumsum = np.cumsum(X, axis=0)

        @njit(fastmath=True)  # type: ignore
        def _cost(min_cost: float, idx: int, t: int) -> float:
            s = 0.0
            n = t - idx

            for j in range(d):
                diff = cumsum[t, j] - cumsum[idx, j]
                p = n / diff
                log_p = np.log(p + 1e-8)
                log_1_p = np.log1p(1e-8 - p)
                s -= n * log_p + (diff - n) * log_1_p

            return s + min_cost

        self.__call__ = _cost


class ExponentialCost(BaseCost):
    def __init__(self, X: np.ndarray):
        """Initialize the cost of a given segment for Exponential data.

        Args:
            X (np.ndarray): The data (must be positive).
        """
        d = X.shape[1]
        cumsum = np.cumsum(X, axis=0)

        @njit(fastmath=True)  # type: ignore
        def _cost(min_cost: float, idx: int, t: int) -> float:
            s = 0.0
            n = t - idx

            for j in range(d):
                diff = cumsum[t, j] - cumsum[idx, j]
                s += n * np.log(diff / n + 1e-8)

            return s + min_cost

        self.__call__ = _cost


class GammaCost(BaseCost):
    def __init__(self, X: np.ndarray, k: float = 1.0):
        """Initialize the cost of a given segment for Gamma data.

        Args:
            X (np.ndarray): The data (must be positive continuous values).
            k (float): The known shape parameter (k > 0).
        """
        d = X.shape[1]
        cumsum = np.cumsum(X, axis=0)
        cumsum_log = np.cumsum(np.log(X + 1e-8), axis=0)

        @njit(fastmath=True)  # type: ignore
        def _cost(min_cost: float, idx: int, t: int) -> float:
            s = 0.0
            n = t - idx

            for j in range(d):
                diff = cumsum[t, j] - cumsum[idx, j]
                diff_log = cumsum_log[t, j] - cumsum_log[idx, j]
                theta = diff / (n * k)
                log_theta = np.log(theta + 1e-8)
                s += (1.0 - k) * diff_log + (n * k * log_theta) + (diff / theta)

            return s + min_cost

        self.__call__ = _cost


class BinomialCost(BaseCost):
    def __init__(self, X: np.ndarray, m: int = 1):
        """Initialize the cost of a given segment for Binomial data.

        Args:
            X (np.ndarray): The data (number of successes).
            m (int): The known number of trials (sample size) per observation.
        """
        d = X.shape[1]
        cumsum = np.cumsum(X, axis=0)

        @njit(fastmath=True)  # type: ignore
        def _cost(min_cost: float, idx: int, t: int) -> float:
            s = 0.0
            n = t - idx
            n_m = n * m

            for j in range(d):
                diff = cumsum[t, j] - cumsum[idx, j]
                p = diff / n_m
                log_p = np.log(p + 1e-8)
                log_1p = np.log1p(1e-8 - p)
                s -= diff * log_p + (n_m - diff) * log_1p

            return s + min_cost

        self.__call__ = _cost


class NegativeBinomialCost(BaseCost):
    def __init__(self, X: np.ndarray, r: int = 1):
        """Initialize the cost of a given segment for Negative Binomial data.

        Args:
            X (np.ndarray): The data (number of trials/successes).
            r (int): The known stopping criterion (e.g., number of failures).
        """
        d = X.shape[1]
        cumsum = np.cumsum(X, axis=0)

        @njit(fastmath=True)  # type: ignore
        def _cost(min_cost: float, idx: int, t: int) -> float:
            s = 0.0
            n = t - idx
            n_r = n * r

            for j in range(d):
                diff = cumsum[t, j] - cumsum[idx, j]
                p = n_r / (n_r + diff)
                log_p = np.log(p + 1e-8)
                log_1p = np.log1p(1e-8 - p)
                s -= n_r * log_p + diff * log_1p

            return s + min_cost

        self.__call__ = _cost
