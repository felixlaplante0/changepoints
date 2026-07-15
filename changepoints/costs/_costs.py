from collections.abc import Callable
from typing import Protocol

import numpy as np
from numba import njit

_LINEAR_TOL = 1e-14


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

    def dust(self, costs: np.ndarray, r: int, s: int, t: int) -> bool:
        """Return whether DUST can safely prune ``s`` using ``r``."""
        ...


def _dust_test(
    statistic: np.ndarray,
    base: np.ndarray,
    minimum: Callable[[float], float],
    mean: Callable[[float], float],
    lower: float,
    upper: float,
):
    """Build the common one-parameter DUST decision test."""

    @njit(fastmath=True)  # type: ignore
    def _test(costs: np.ndarray, r: int, s: int, t: int) -> bool:
        qt, qs, qr = costs[t], costs[s], costs[r]
        if base.size:
            qt += base[0] - base[t]
            qs += base[0] - base[s]
            qr += base[0] - base[r]
        a = (statistic[t] - statistic[s]) / (t - s)
        b = (statistic[s] - statistic[r]) / (s - r)
        c = (qt - qs) / (t - s)
        d = (qs - qr) / (s - r)
        ds = a - b
        dq = c - d
        if minimum(a) - c > 0.0:
            return True
        if abs(ds) < _LINEAR_TOL:
            return dq < 0.0

        theta = -dq / ds
        m = mean(theta)
        x = (m - a) / ds
        if np.isfinite(x) and x > 0.0 and lower <= m <= upper:
            return minimum(m) - c - x * dq > 0.0

        boundary = upper if ds > 0.0 else lower
        if np.isfinite(boundary):
            x = (boundary - a) / ds
            return x > 0.0 and minimum(boundary) - c - x * dq > 0.0

        # The positive-domain models have logarithmic upper tails. If their
        # unconstrained maximizer is at infinity, the decision diverges.
        return ds > 0.0 and dq <= 0.0

    return _test


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

        @njit(fastmath=True)  # type: ignore
        def _minimum(x: float) -> float:
            return -(x * x)

        @njit(fastmath=True)  # type: ignore
        def _mean(theta: float) -> float:
            return theta / 2.0

        self.dust = _dust_test(
            cumsum[:, 0], cumsum2[:, 0], _minimum, _mean, -np.inf, np.inf
        )


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

        @njit(fastmath=True)  # type: ignore
        def _minimum(x: float) -> float:
            return np.log(x + 1e-8) + 1.0

        @njit(fastmath=True)  # type: ignore
        def _mean(theta: float) -> float:
            return -1.0 / theta if theta < 0.0 else np.nan

        self.dust = _dust_test(
            cumsum2[:, 0], np.empty(0), _minimum, _mean, 0.0, np.inf
        )


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

        @njit(fastmath=True)  # type: ignore
        def _minimum(x: float) -> float:
            return x - x * np.log(x + 1e-8)

        @njit(fastmath=True)  # type: ignore
        def _mean(theta: float) -> float:
            return np.exp(theta)

        self.dust = _dust_test(
            cumsum[:, 0], np.empty(0), _minimum, _mean, 0.0, np.inf
        )


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

        @njit(fastmath=True)  # type: ignore
        def _minimum(x: float) -> float:
            p = 1.0 / x
            return -np.log(p + 1e-8) - (x - 1.0) * np.log1p(1e-8 - p)

        @njit(fastmath=True)  # type: ignore
        def _mean(theta: float) -> float:
            return 1.0 / (1.0 - np.exp(theta)) if theta < 0.0 else np.nan

        self.dust = _dust_test(
            cumsum[:, 0], np.empty(0), _minimum, _mean, 1.0, np.inf
        )


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

        @njit(fastmath=True)  # type: ignore
        def _minimum(x: float) -> float:
            return np.log(x + 1e-8)

        @njit(fastmath=True)  # type: ignore
        def _mean(theta: float) -> float:
            return -1.0 / theta if theta < 0.0 else np.nan

        self.dust = _dust_test(
            cumsum[:, 0], np.empty(0), _minimum, _mean, 0.0, np.inf
        )


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

        @njit(fastmath=True)  # type: ignore
        def _minimum(x: float) -> float:
            return k * np.log(x / k + 1e-8) + k

        @njit(fastmath=True)  # type: ignore
        def _mean(theta: float) -> float:
            return -k / theta if theta < 0.0 else np.nan

        self.dust = _dust_test(
            cumsum[:, 0],
            (1.0 - k) * cumsum_log[:, 0],
            _minimum,
            _mean,
            0.0,
            np.inf,
        )


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

        @njit(fastmath=True)  # type: ignore
        def _minimum(x: float) -> float:
            p = x / m
            return -x * np.log(p + 1e-8) - (m - x) * np.log1p(1e-8 - p)

        @njit(fastmath=True)  # type: ignore
        def _mean(theta: float) -> float:
            if theta >= 0.0:
                z = np.exp(-theta)
                return m / (1.0 + z)
            z = np.exp(theta)
            return m * z / (1.0 + z)

        self.dust = _dust_test(
            cumsum[:, 0], np.empty(0), _minimum, _mean, 0.0, float(m)
        )


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

        @njit(fastmath=True)  # type: ignore
        def _minimum(x: float) -> float:
            p = r / (r + x)
            return -r * np.log(p + 1e-8) - x * np.log1p(1e-8 - p)

        @njit(fastmath=True)  # type: ignore
        def _mean(theta: float) -> float:
            if theta >= 0.0:
                return np.nan
            z = np.exp(theta)
            return r * z / (1.0 - z)

        self.dust = _dust_test(
            cumsum[:, 0], np.empty(0), _minimum, _mean, 0.0, np.inf
        )
