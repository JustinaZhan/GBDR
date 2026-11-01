# GBDR

GBDR (Granular-ball Density Ratio) is a weakly supervised anomaly detector. A granular-ball tree defines continuous radial functions from unlabeled samples, and a strongly convex density ratio objective learns their coefficients using a small set of labeled anomalies.

`gbdr.py` contains the core algorithm. Dependencies: NumPy, SciPy, and scikit-learn.

## Acknowledgment

We gratefully acknowledge [Professor Mingjie Cai](https://github.com/HNU-MingjieCai), our advisor, for his guidance and support throughout this work.

## Usage

```python
from gbdr import GBDR

# U: unlabeled samples; S: labeled anomalies; X: samples to score.
model = GBDR(random_state=0).fit(U, S)
scores = model.decision_function(X)  # Higher scores indicate anomalies.
```

Inputs are finite, two-dimensional numerical attribute matrices with the same preprocessing. Exclude the labeled anomalies in `S` from `U`. By default, the regularization parameter is selected from `{0.1, 1, 10}` using the labeled anomalies; one labeled anomaly uses `1`.


