import numpy as np
import numpy.typing as npt


def validate_X(X: npt.ArrayLike) -> np.ndarray:
    """Validate the input data.

    Args:
        X (npt.ArrayLike): The input data.

    Raises:
        ValueError: If X is not a 2D array.
        ValueError: If X contains NaN or Inf.

    Returns:
        np.ndarray: The validated data.
    """
    X = np.asarray(X)  # type: ignore

    if X.ndim != 2:  # noqa: PLR2004
        raise ValueError("X must be a 2D array")

    if np.isnan(X).any():
        raise ValueError("X must not contain NaN")
    if np.isinf(X).any():
        raise ValueError("X must not contain Inf")

    return X
