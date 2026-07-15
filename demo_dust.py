"""Compare DUST runtime with and without Numba compilation."""

import subprocess
import tempfile
from time import perf_counter

import numpy as np

from changepoints import DUST, PELT
from changepoints.costs import GaussianMeanCost


def _benchmark(model, signal):
    start = perf_counter()
    first = model.fit_predict(signal).copy()
    compiled = perf_counter() - start

    start = perf_counter()
    second = model.fit_predict(signal)
    cached = perf_counter() - start
    np.testing.assert_array_equal(first, second)
    return compiled, cached, second


def _benchmark_r(signal):
    with tempfile.NamedTemporaryFile() as data:
        np.savetxt(data.name, signal[:, 0])
        code = """
        library(dust)
        x <- scan(commandArgs(TRUE)[1], quiet=TRUE)
        elapsed <- system.time(fit <- dust.1D(x, penalty=15, model="gauss"))[3]
        cat(sprintf("vrunge/dust: %.4f s\\nChangepoints: %s\\n", elapsed,
                    paste(fit$changepoints, collapse=" ")))
        """
        return subprocess.run(  # noqa: S603
            ["Rscript", "-e", code, data.name],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
        ).stdout


def main():  # noqa: D103
    rng = np.random.default_rng(42)
    n = 100_000
    signal = np.empty((n, 1))
    for i, start in enumerate(range(0, n, 20_000)):
        signal[start : start + 20_000, 0] = rng.normal(2 * (i % 2), 1, 20_000)

    for name, model in (
        ("DUST", DUST(GaussianMeanCost, penalty=30)),
        ("PELT", PELT(GaussianMeanCost, penalty=30)),
    ):
        compiled, cached, changepoints = _benchmark(model, signal)
        print(f"{name} with compilation:    {compiled:.4f} s")  # noqa: T201
        print(f"{name} without compilation: {cached:.4f} s")  # noqa: T201
        print(f"{name} changepoints: {changepoints}\n")  # noqa: T201
    print(_benchmark_r(signal))  # noqa: T201


if __name__ == "__main__":
    main()
