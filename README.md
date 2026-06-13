# On the Nearest Special Unitary Matrix

Reference implementation for the paper [**"On the Nearest Special Unitary Matrix"**](https://www.algorhythmandflow.com/research/nearest_su.pdf) by Akshay Chandrasekhar 🌐🔢🧮⚛️

Computes the nearest $SU(n)$ matrix (in the Frobenius norm sense) to an arbitrary $n \times n$ complex matrix.

For a simpler overview of the paper, see the [explainer site](https://akschion.github.io/Nearest_SU/).

## Methods

| Method | Best for | Notes |
|---|---|---|
| `su2_algebraic` | $n = 2$ | Closed-form and fastest; uses $M$ directly |
| `nearest_su` (numpy) | $n = 3$ | Analytical solution with reasonable runtime; unstable for $n>3$ |
| `nearest_su` (mpmath) | $3 \leq n \leq 7$ | Analytical with higher precision; very slow |
| `numerical_optimization` | $n \geq 3$ | **Recommended** — fast and numerically stable |

> **Note:** All methods except `su2_algebraic` expect the SVD of the input matrix to be precomputed and passed in as arguments.


## Requirements

- Python 3.x
- NumPy
- SciPy
- [mpmath](https://mpmath.org/) *(required only for the high-precision analytical method)*
