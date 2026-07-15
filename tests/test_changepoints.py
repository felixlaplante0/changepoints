"""Tests for the changepoints package."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from changepoints import DUST
from changepoints.costs import (
    BinomialCost,
    ExponentialCost,
    GammaCost,
    GaussianMeanCost,
    GaussianMeanVarianceCost,
    GaussianVarianceCost,
    GeometricCost,
    NegativeBinomialCost,
    PoissonCost,
)


def _optimal_partition(X, cost, penalty, **kwargs):
    segment_cost = cost(X, **kwargs).__call__
    n = X.shape[0]
    costs = np.zeros(n)
    previous = np.zeros(n, dtype=int)
    for t in range(1, n):
        values = np.array([segment_cost(costs[s], s, t) for s in range(t)])
        previous[t] = np.argmin(values)
        costs[t] = values[previous[t]] + penalty

    changepoints = []
    t = n - 1
    while t > 0:
        t = previous[t]
        if t > 0:
            changepoints.append(t)
    return np.array(changepoints[::-1]), costs


@pytest.mark.parametrize(
    ("cost", "values", "kwargs"),
    [
        (GaussianMeanCost, [0.0] * 12 + [4.0] * 12, {}),
        (GaussianVarianceCost, [1.0] * 12 + [4.0] * 12, {}),
        (PoissonCost, [0.0] * 12 + [5.0] * 12, {}),
        (GeometricCost, [1.0] * 12 + [5.0] * 12, {}),
        (ExponentialCost, [0.5] * 12 + [4.0] * 12, {}),
        (GammaCost, [1.0] * 12 + [6.0] * 12, {"k": 2.0}),
        (BinomialCost, [0.0] * 12 + [5.0] * 12, {"m": 5}),
        (NegativeBinomialCost, [0.0] * 12 + [6.0] * 12, {"r": 3}),
    ],
)
def test_exact(cost, values, kwargs):
    """DUST matches unpruned optimal partitioning for every supported cost."""
    X = np.asarray(values)[:, None]
    expected, expected_costs = _optimal_partition(X, cost, 2.0, **kwargs)
    model = DUST(cost, 2.0, **kwargs).fit(X)
    np.testing.assert_array_equal(model.chgpts, expected)
    np.testing.assert_allclose(model.min_costs, expected_costs)


def test_bad_dimension():
    """DUST rejects multivariate signals."""
    with pytest.raises(ValueError, match="exactly one column"):
        DUST().fit(np.ones((10, 2)))


def test_bad_cost():
    """DUST rejects costs outside the one-parameter family."""
    with pytest.raises(ValueError, match="one-parameter"):
        DUST(GaussianMeanVarianceCost).fit(np.ones((10, 1)))


def test_predict_unfit():
    """Prediction requires a fitted estimator."""
    with pytest.raises(NotFittedError):
        DUST().predict()
