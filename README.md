# Change Point Detection with PELT

This repository demonstrates the use of the **Pruned Exact Linear Time (PELT)** algorithm for detecting change points in a signal.

---

## Installation

You can install the required dependencies using `pip`:

```bash
pip install numpy matplotlib changepoints
```

## Usage

Here is a simple example. It uses a Gaussian mean cost function to detect shifts in the mean of a synthetic time series, but many other costs are defined.

```python
import itertools
import time

import matplotlib.pyplot as plt
import numpy as np

from changepoints import PELT
from changepoints.costs import GaussianMeanCost

n_samples = 100000
signal = np.zeros((n_samples, 1))
breakpoints = [0, 20000, 40000, 60000, 80000, 100000]

for m, (i, j) in enumerate(itertools.pairwise(breakpoints)):
    mean = 0 if m % 2 == 0 else 2
    signal[i:j] = np.random.normal(mean, 1, size=(j - i, 1))

model = PELT(GaussianMeanCost, 30)

start = time.time()
model.fit(signal)
end = time.time()
print(f"First run (compilation): {end - start:.4f} s")  # 1.6 seconds

start = time.time()
model.fit_predict(signal)
end = time.time()
print(f"Second run (cached): {end - start:.4f} s")  # 0.8 second

plt.plot(signal)
for i in model.chgpts:
    plt.axvline(x=i, color="r")
```
plt.show()

```
