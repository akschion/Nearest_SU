"""
Code for "On the Nearest Special Unitary Matrix" by Akshay Chandrasekhar

Performs benchmarking experiments of proposed algorithms against heuristic
solutions. See bottom for run commands.

Copyright © 2026 Akshay Chandrasekhar

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the “Software”),
to deal in the Software without restriction, including without
limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons
to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.optimize import brentq

try:
    import mpmath as mp
except ModuleNotFoundError:
    mp = None

########################################Settings and constants

TOL = 1e-10
SINGULAR_VALUE_RTOL = 1e-12
SU_VALIDITY_TOL = 1e-8
NEAREST_EPS = 1e-8
SU2_DEGENERATE_RTOL = 1e-7
MATRIX_VALUE_MAX = 2
PHASE_CANDIDATE_TOL = 1e-8
UNIT_NORM_TOL = 1e-7
NUMPY_ROOT_IMAG_TOL = 1e-6
NUMERIC_X_TOL = 1e-12
NUMERIC_GRID_SIZE = 16
NUMERIC_GRID_REFINEMENTS = 2
NUMERIC_MAX_ITER = 50

MP_DPS = 100
MP_EPS = None if mp is None else mp.mpf(2) ** (-1024)
MP_BIG = None if mp is None else mp.mpf(2) ** 8191
MP_SMALL = None if mp is None else mp.mpf(2) ** (-8191)
MP_REAL_TOL = None if mp is None else mp.mpf(2) ** (-128)
MP_ROOT_MAXSTEPS = 1000
MP_ROOT_EXTRAPREC = MP_DPS
mp_zero = None if mp is None else mp.mpf("0")
mp_one = None if mp is None else mp.mpf("1")
eps_mpmath = MP_EPS

########################################Testing functions and utilities

def normalize_precision(precision):
    if precision == "numpy":
        return "np"
    if precision not in ("np", "mp"):
        raise ValueError("precision must be either 'np' or 'mp'")
    return precision


def singular_value_tol(s):
    values = np.asarray(s, dtype=float)
    if values.size == 0:
        return SINGULAR_VALUE_RTOL
    return SINGULAR_VALUE_RTOL * max(1.0, float(np.max(np.abs(values))))


def H(M):
    return np.conjugate(M).T


def svd_phase(U, Vh):
    phase = np.linalg.det(U) * np.linalg.det(Vh)
    return phase / abs(phase)


def reconstruct(U, s, Vh):
    return (U * s) @ Vh


def make_test_matrix(args, rng):
    M = rng.uniform(-MATRIX_VALUE_MAX, MATRIX_VALUE_MAX, size=(args.n, args.n))
    M = M + 1j * rng.uniform(-MATRIX_VALUE_MAX, MATRIX_VALUE_MAX, size=(args.n, args.n))

    if not (args.real_determinant or args.singular or args.repeated_singular_values):
        return M

    U, s, Vh = np.linalg.svd(M, full_matrices=True)

    if args.repeated_singular_values and args.n >= 2:
        idx = rng.choice(np.arange(args.n - 1))
        s[idx + 1] = s[idx]

    if args.singular:
        zero_count = min(max(1, args.zero_singular_values), args.n - 1)
        s[-zero_count:] = 0.0

    if args.real_determinant:
        phase = svd_phase(U, Vh)
        target_phase = 1.0 if rng.random() < 0.5 else -1.0
        U[:, -1] *= target_phase / phase

    return reconstruct(U, s, Vh)


def verify_su(Q):
    I = np.eye(Q.shape[0], dtype=complex)
    unitary_tol = SU_VALIDITY_TOL * max(1.0, np.sqrt(Q.shape[0]))
    return (np.allclose(H(Q) @ Q, I, atol=unitary_tol, rtol=unitary_tol) and #unitary check
            np.allclose(np.linalg.det(Q), 1.0, atol=SU_VALIDITY_TOL, rtol=SU_VALIDITY_TOL)) #determinant check

########################################Polynomial Construction and Root-Finding

def build_univariate_polynomial(s, phase, precision="np"):
    """Return coefficients and backsubstitution data for the nonreal phase case.

    Coefficients are ascending: coeff[k] multiplies b_n^k.
    """

    precision = normalize_precision(precision)
    if precision == "mp":
        if mp is None:
            raise ModuleNotFoundError("mpmath is required for --nearest-su-precision mp")
        scalar = mp.mpf
    else:
        scalar = float

    n = len(s)
    bn = n - 1
    if precision == "mp":
        u = [scalar(str(s[-1] / value)) for value in s]
        rho_re = scalar(str(np.real(phase)))
        rho_im = scalar(str(np.imag(phase)))
    else:
        u = [float(s[-1] / value) for value in s]
        rho_re = float(np.real(phase))
        rho_im = float(np.imag(phase))
    zero = scalar("0") if precision == "mp" else 0.0
    one = scalar("1") if precision == "mp" else 1.0

    def clean(P, tol=0.0):
        if precision == "mp":
            return {e: c for e, c in P.items() if mp.fabs(c) > mp.mpf(str(tol))}
        return {e: c for e, c in P.items() if abs(float(c)) > tol}

    def const(c):
        return {} if c == zero else {tuple([0] * n): c}

    def var(i, c=one):
        e = [0] * n
        e[i] = 1
        return {tuple(e): c}

    def add(A, B, scale=one):
        C = dict(A)
        for e, c in B.items():
            C[e] = C.get(e, zero) + scale * c
        return clean(C)

    def mul(A, B):
        C = {}
        for ea, ca in A.items():
            for eb, cb in B.items():
                e = tuple(a + b for a, b in zip(ea, eb))
                C[e] = C.get(e, zero) + ca * cb
        return clean(C)

    def poly_pow(P, power):
        out = const(one)
        for _ in range(power):
            out = mul(out, P)
        return out

    def scale(P, c):
        return clean({e: c * value for e, value in P.items()})

    def complex_mul(z, w):
        real = add(mul(z[0], w[0]), mul(z[1], w[1]), -one)
        imag = add(mul(z[0], w[1]), mul(z[1], w[0]))
        return real, imag

    def reduce_var(P, i, relation):
        out = {}
        for e, c in P.items():
            power = e[i]
            base = list(e)
            base[i] = 0
            term = mul({tuple(base): c}, poly_pow(relation, power // 2))
            if power % 2:
                term = mul(term, var(i))
            out = add(out, term)
        return out

    def split_affine(P, i):
        c0, c1 = {}, {}
        for e, c in P.items():
            power = e[i]
            base = list(e)
            base[i] = 0
            base = tuple(base)
            if power == 0:
                c0[base] = c0.get(base, zero) + c
            elif power == 1:
                c1[base] = c1.get(base, zero) + c
            else:
                raise ValueError(f"polynomial is not affine in variable {i}")
        return clean(c1), clean(c0)

    def univariate_coeffs(P, i):
        degree = 0
        for e in P:
            if any(power and j != i for j, power in enumerate(e)):
                raise ValueError("polynomial is not univariate")
            degree = max(degree, e[i])

        coeffs = [zero for _ in range(degree + 1)]
        for e, c in P.items():
            coeffs[e[i]] += c

        trim_tol = TOL if precision == "np" else 0.0
        while len(coeffs) > 1:
            leading = coeffs[-1]
            if precision == "mp":
                keep = mp.fabs(leading) > mp.mpf(str(trim_tol))
            else:
                keep = abs(float(leading)) > trim_tol
            if keep:
                break
            coeffs.pop()
        return coeffs

    # det constraint:
    # b_n = Im(rho * conjugate(prod_{i<n} z_i)).
    product = (const(one), const(zero))
    for i in range(n - 1):
        product = complex_mul(product, (var(i), scale(var(bn), -u[i])))

    equation = add(scale(product[1], rho_re), scale(product[0], rho_im))
    equation = add(equation, var(bn), -one)
    frames = []

    for i in range(n - 1):
        relation = add(const(one), scale(poly_pow(var(bn), 2), -(u[i] ** 2)))
        reduced = reduce_var(equation, i, relation)
        c1, c0 = split_affine(reduced, i)
        equation = add(mul(mul(c1, c1), relation), mul(c0, c0), -one)
        frames.append((i, c1, c0))

    def eval_poly(P, values):
        total = zero
        for e, c in P.items():
            term = c
            for i, power in enumerate(e):
                if power:
                    term *= values[i] ** power
            total += term
        return total

    def square_relation(i, root):
        return one - (u[i] * root) ** 2

    return univariate_coeffs(equation, bn), frames, eval_poly, square_relation


# Aberth solver translated in the earlier scratch implementation.
def ctest(a, il, i, ir):
    toler = mp.mpf("0.4")
    s1 = (a[i] - a[il]) * (ir - i)
    s2 = (a[ir] - a[i]) * (i - il)
    return s1 > (s2 + toler)


def left(h, i):
    for il in range(i - 1, -1, -1):
        if h[il]:
            return il
    return None


def right(n, h, i):
    for ir in range(i + 1, n):
        if h[ir]:
            return ir
    return None


def cmerge(n, a, i, m, h):
    i_py = i - 1
    il = left(h, i_py)
    ir = right(n, h, i_py)
    if il is None or ir is None:
        return
    if ctest(a, il, i_py, ir):
        return
    h[i_py] = False
    while True:
        if il == (i_py - m):
            tstl = True
        else:
            ill = left(h, il)
            tstl = ctest(a, ill, il, ir) if ill is not None else True
        if ir == min(n - 1, i_py + m):
            tstr = True
        else:
            irr = right(n, h, ir)
            tstr = ctest(a, il, ir, irr) if irr is not None else True
        h[il] = tstl
        h[ir] = tstr
        if tstl and tstr:
            return
        if not tstl:
            il = ill
        if not tstr:
            ir = irr


def cnvex(n, a, h):
    for i in range(n):
        h[i] = True
    k = int(mp.log(n - 2) / mp.log(2)) if n - 2 > 0 else 0
    if 2 ** (k + 1) <= (n - 2):
        k += 1
    m = 1
    for _ in range(0, k + 1):
        nj = max(0, (n - 2 - m) // (2 * m))
        for j in range(0, nj + 1):
            cmerge(n, a, (2 * j + 1) * m + 1, m, h)
        m += m


def start(n, a, small, big):
    for i in range(n + 1):
        a[i] = mp.log(a[i]) if a[i] != 0 else mp.mpf("-1e30")
    h = [True] * (n + 1)
    cnvex(n + 1, a, h)
    pi2 = 2 * mp.pi
    sigma = mp.mpf("0.7")
    y = [None] * n
    radius = [None] * n
    iold = 0
    th = pi2 / n
    nz = 0
    for i in range(1, n + 1):
        if h[i]:
            nzeros = i - iold
            temp = (a[iold] - a[i]) / nzeros
            xsmall = mp.log(small)
            xbig = mp.log(big)
            if (temp < -xbig) and (temp >= xsmall):
                nz += nzeros
                r = mp_one / mp.mpf(big)
            elif temp < xsmall:
                nz += nzeros
                r = mp.mpf("0")
            elif temp > xbig:
                nz += nzeros
                r = mp.mpf(big)
            else:
                r = mp.exp(temp)
            ang = pi2 / nzeros
            for j in range(iold, i):
                jj = j - iold + 1
                radius[j] = -1 if (r <= mp_one / mp.mpf(big)) or (r == mp.mpf(big)) else r
                angle = ang * jj + th * (i + 1) + sigma
                y[j] = r * mp.mpc(mp.cos(angle), mp.sin(angle))
            iold = i
    return y, radius, nz


def newton(n, poly, apoly, apolyr, z, small):
    az = mp.fabs(z)
    if az <= 1:
        p = poly[n]
        p1 = p
        ap = apoly[n]
        for i in range(n, 1, -1):
            p = p * z + poly[i - 1]
            p1 = p1 * z + p
            ap = ap * az + apoly[i - 1]
        p = p * z + poly[0]
        ap = ap * az + apoly[0]
        corr = p / p1
        absp = mp.fabs(p)
        ok = absp > (small + ap)
        if not ok:
            return corr, n * (absp + ap) / mp.fabs(p1), ok
        return corr, None, ok

    zi = mp_one / z
    azi = mp_one / az
    p = poly[0]
    p1 = p
    ap = apolyr[n]
    for i in range(n, 1, -1):
        p = p * zi + poly[n - i + 1]
        p1 = p1 * zi + p
        ap = ap * azi + apolyr[i - 1]
    p = p * zi + poly[n]
    ap = ap * azi + apolyr[0]
    absp = mp.fabs(p)
    ok = absp > (small + ap)
    ppsp = (p * z) / p1
    den = n * ppsp - mp_one
    corr = z * (ppsp / den)
    if ok:
        return corr, None, ok
    radius = mp.fabs(ppsp) + (ap * az) / mp.fabs(p1)
    radius = n * radius / mp.fabs(den)
    return corr, radius * az, ok


def aberth(n, j, root):
    abcorr = 0
    zj = root[j]
    for i in range(n):
        if i != j:
            abcorr += mp_one / (zj - root[i])
    return abcorr


def polzeros(poly, eps, big, small, nitmax):
    n = len(poly) - 1
    while n > 0 and mp.fabs(poly[n]) < eps_mpmath:
        poly = poly[:-1]
        n -= 1
    add_zero = False
    while n > 0 and mp.fabs(poly[0]) < eps_mpmath:
        add_zero = True
        poly = poly[1:]
        n -= 1
    if n <= 0:
        return ([mp_zero] if add_zero else []), ([1] if add_zero else []), ([True] if add_zero else []), 0

    apoly = [mp.fabs(poly[i]) for i in range(n + 1)]
    apolyr = apoly.copy()
    root, radius_start, _ = start(n, apolyr.copy(), small, big)
    err = [radius_start[i] != -1 for i in range(n)]

    for k in range(n + 1):
        apolyr[n - k] = eps * apoly[k] * (mp.mpf("3.8") * (n - k) + 1)
        apoly[k] = eps * apoly[k] * (mp.mpf("3.8") * k + 1)

    nzeros = 0
    for iteration in range(1, nitmax + 1):
        for i in range(n):
            if err[i]:
                corr, _, ok = newton(n, poly, apoly, apolyr, root[i], small)
                if ok:
                    abcorr = aberth(n, i, root)
                    root[i] -= corr / (mp_one - corr * abcorr)
                else:
                    err[i] = False
                    nzeros += 1
                    if nzeros == n:
                        if add_zero:
                            root.append(mp_zero)
                            radius_start.append(1)
                            err.append(True)
                        return root, radius_start, err, iteration
    if add_zero:
        root.append(mp_zero)
        radius_start.append(1)
        err.append(True)
    return root, radius_start, err, nitmax


def aberth_roots(coeffs, mp_dps=MP_DPS):
    if mp is None:
        raise ModuleNotFoundError("mpmath is required for --nearest-su-precision mp")
    mp.mp.dps = mp_dps
    roots, _, _, _ = polzeros(
        [mp.mpc(c) for c in coeffs],
        MP_EPS,
        MP_BIG,
        MP_SMALL,
        MP_ROOT_MAXSTEPS,
    )
    return roots


def solve_real_roots(coeffs, precision="np", mp_dps=MP_DPS):
    precision = normalize_precision(precision)
    if len(coeffs) <= 1:
        return []

    if precision == "mp":
        if mp is None:
            raise ModuleNotFoundError("mpmath is required for --nearest-su-precision mp")
        mp.mp.dps = mp_dps
        roots = aberth_roots(coeffs, mp_dps)
        real_tol = max(MP_REAL_TOL, mp.mpf(10) ** (-(mp_dps // 2)))
        roots = [mp.re(r) for r in roots if mp.fabs(mp.im(r)) < real_tol]
        tol = real_tol
    else:
        roots = np.roots(np.array(list(reversed(coeffs)), dtype=float))
        roots = [float(np.real(r)) for r in roots if abs(np.imag(r)) < NUMPY_ROOT_IMAG_TOL]
        derivative = [k * c for k, c in enumerate(coeffs)][1:]

        def eval1(poly, x):
            value = poly[-1]
            for c in reversed(poly[:-1]):
                value = value * x + c
            return value

        polished = []
        for r in roots:
            for _ in range(12):
                slope = eval1(derivative, r)
                if abs(slope) < TOL:
                    break
                step = eval1(coeffs, r) / slope
                r -= step
                if abs(step) < TOL:
                    break
            polished.append(r)
        roots = polished
        tol = TOL

    unique = []
    for r in roots:
        if all(abs(float(r - old)) > float(tol) for old in unique):
            unique.append(r)
    return unique


def solve_bn_roots(coeffs, n, phase, precision="np", mp_dps=MP_DPS):
    precision = normalize_precision(precision)
    real_phase = abs(np.imag(phase)) < PHASE_CANDIDATE_TOL
    sign = 1.0 if np.imag(phase) > 0 or (real_phase and np.real(phase) < 0.0) else -1.0
    bn_max = abs(np.imag(phase)) if np.real(phase) > 0.0 else 1.0

    if n % 2 == 0:
        reduced_coeffs = coeffs[::2]
        roots_sq = solve_real_roots(reduced_coeffs, precision, mp_dps)
        roots = []
        for root_sq in roots_sq:
            if float(root_sq) < -TOL:
                continue
            if precision == "mp":
                root = mp.sqrt(max(root_sq, mp_zero))
            else:
                root = np.sqrt(max(float(root_sq), 0.0))
            if real_phase and np.real(phase) < 0.0 and abs(float(root)) > TOL:
                roots += [root, -root]
            else:
                roots.append(root if sign > 0 else -root)
        return [root for root in roots if abs(float(root)) <= bn_max + TOL]

    roots = solve_real_roots(coeffs, precision, mp_dps)
    if real_phase:
        return [root for root in roots if abs(float(root)) <= bn_max + TOL]
    return [
        root for root in roots
        if np.sign(float(root)) == np.sign(np.imag(phase)) and abs(float(root)) <= bn_max + TOL
    ]

########################################Solutions

def naive_1(U, s, Vh):
    Q = U @ Vh
    return np.conjugate(np.linalg.det(Q) ** (1.0 / Q.shape[0])) * Q


def naive_2(U, s, Vh):
    phase = svd_phase(U, Vh)
    Up = U.copy()
    Up[:, -1] *= np.conjugate(phase)
    return Up @ Vh


def su2_linear_algebra(U, s, Vh):
    if U.shape[0] != 2:
        raise ValueError("SU(2) closed form only applies to n=2")

    phase = svd_phase(U, Vh)
    s1, s2 = s
    z1 = s1 + s2 * phase
    z2 = s1 * phase + s2
    norm = np.sqrt(max(s1 * s1 + 2.0 * s1 * s2 * np.real(phase) + s2 * s2, 0.0))
    degenerate_tol = SU2_DEGENERATE_RTOL * max(1.0, np.sqrt(s1 * s1 + s2 * s2))

    if norm < degenerate_tol:
        z = np.array([1.0, -1.0], dtype=complex)
    else:
        z = np.array([np.conjugate(z1 / norm), np.conjugate(z2 / norm)])

    return (U * z) @ Vh


def su2_algebraic(M):
    if M.shape[0] != 2:
        raise ValueError("SU(2) algebraic formula only applies to n=2")

    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    frob_sq = float(np.real(np.vdot(M, M)))
    denominator = np.sqrt(max(frob_sq + 2.0 * np.real(det), 0.0))
    degenerate_tol = SU2_DEGENERATE_RTOL * max(1.0, np.sqrt(frob_sq))
    if denominator < degenerate_tol:
        U, s, Vh = np.linalg.svd(M, full_matrices=True)
        return su2_linear_algebra(U, s, Vh)

    return np.array([[M[0, 0] + np.conjugate(M[1, 1]), M[0, 1] - np.conjugate(M[1, 0])], 
                     [M[1, 0] - np.conjugate(M[0, 1]), M[1, 1] + np.conjugate(M[0, 0])]
                     ], dtype=np.complex128) / denominator


def nearest_su(U, s, Vh, precision="np", mp_dps=MP_DPS):
    precision = normalize_precision(precision)
    n = len(s)
    phase = svd_phase(U, Vh)

    if n == 1:
        return np.array([[1. + 0j]], dtype=np.complex128)
    if n == 2:
        return su2_linear_algebra(U, s, Vh)

    real_phase = abs(np.imag(phase)) < PHASE_CANDIDATE_TOL
    if s[-1] < singular_value_tol(s) or (real_phase and np.real(phase) > 0.0):
        return naive_2(U, s, Vh)

    if precision == "mp":
        if mp is None:
            raise ModuleNotFoundError("mpmath is required for --nearest-su-precision mp")
        mp.mp.dps = mp_dps
        scalar = mp.mpf
    else:
        scalar = float

    coeffs, frames, eval_poly, square_relation = build_univariate_polynomial(s, phase, precision)
    roots = solve_bn_roots(coeffs, n, phase, precision, mp_dps)
    if precision == "mp":
        u_ratios = [scalar(str(s[-1] / value)) for value in s]
    else:
        u_ratios = [float(s[-1] / value) for value in s]
    bn = n - 1

    best_z = None
    best_gain = -np.inf

    for root in roots:
        if abs(float(root)) > 1.0 + TOL:
            continue

        partial_solutions = [{bn: root}]
        for i, c1, c0 in reversed(frames):
            next_solutions = []
            for sol in partial_solutions:
                c1_val = eval_poly(c1, sol)
                c0_val = eval_poly(c0, sol)
                relation_val = square_relation(i, sol[bn])

                if abs(float(c1_val)) <= TOL: #this changed from paper reporting which improved analytical MP (used to be much tighter)
                    if float(relation_val) < -TOL:
                        continue
                    magnitude = mp.sqrt(max(relation_val, mp_zero)) if precision == "mp" else np.sqrt(max(float(relation_val), 0.0))
                    branch = dict(sol)
                    branch[i] = magnitude
                    next_solutions.append(branch)
                else:
                    branch = dict(sol)
                    branch[i] = -c0_val / c1_val
                    next_solutions.append(branch)
            partial_solutions = next_solutions

        for sol in partial_solutions:
            b = np.array([float(u_ratios[i] * root) for i in range(n)])
            a = np.array([float(sol[i]) for i in range(n - 1)])
            prefix = np.prod(a + 1j * b[:-1])
            a = np.append(a, float(np.real(phase * np.conjugate(prefix))))

            z = a + 1j * b
            if not np.all(np.isfinite(z)):
                continue
            if not np.allclose(np.abs(z), 1.0, atol=UNIT_NORM_TOL, rtol=UNIT_NORM_TOL):
                continue
            if not np.allclose(np.prod(z), phase, atol=UNIT_NORM_TOL, rtol=UNIT_NORM_TOL):
                continue
            # After validation, remove tiny roundoff so the reported matrix is SU.
            z = z / np.abs(z)
            z[-1] = phase / np.prod(z[:-1])

            gain = float(np.dot(s, np.real(z)))
            if gain > best_gain:
                best_gain = gain
                best_z = z

    if best_z is None:
        raise ValueError("no feasible stationary candidates found")

    return (U * np.conjugate(best_z)) @ Vh


def numerical_optimization(U, s, Vh):
    n = len(s)
    phase = svd_phase(U, Vh)

    if n == 1:
        return np.array([[1. + 0j]], dtype=np.complex128)
    if s[-1] < singular_value_tol(s):
        return naive_2(U, s, Vh)

    real_phase = abs(np.imag(phase)) < PHASE_CANDIDATE_TOL
    if real_phase and np.real(phase) > 0.0:
        return naive_2(U, s, Vh)

    real_negative_phase = real_phase and np.real(phase) < 0.0
    sign = 1.0 if real_negative_phase or np.imag(phase) > 0 else -1.0
    bn_max = abs(np.imag(phase)) if np.real(phase) > 0.0 else 1.0
    x_max = min(1.0, float((s[-1] / s[0]) * bn_max))

    ratios = s[0] / s[:-1]
    prefix_s = s[:-1]
    phase_angle = float(np.angle(phase))
    slope = (s[0] / s[-1]) * sign

    def residual_scalar(x):
        b_prefix = np.clip(float(x) * ratios, 0.0, 1.0)
        theta_n = phase_angle - sign * float(np.sum(np.arcsin(b_prefix)))
        return slope * x - np.sin(theta_n)

    def residual_grid(xs):
        b_prefix = np.clip(xs[:, None] * ratios[None, :], 0.0, 1.0)
        theta_n = phase_angle - sign * np.sum(np.arcsin(b_prefix), axis=1)
        return slope * xs - np.sin(theta_n)

    def score(x):
        b_prefix = np.clip(float(x) * ratios, 0.0, 1.0)
        a_prefix = np.sqrt(np.maximum((1.0 + b_prefix) * (1.0 - b_prefix), 0.0))
        theta_n = phase_angle - sign * float(np.sum(np.arcsin(b_prefix)))
        return float(prefix_s @ a_prefix + s[-1] * np.cos(theta_n))

    def construct_z(x):
        b_prefix = np.clip(float(x) * ratios, 0.0, 1.0)
        a_prefix = np.sqrt(np.maximum((1.0 + b_prefix) * (1.0 - b_prefix), 0.0))
        theta_n = phase_angle - sign * float(np.sum(np.arcsin(b_prefix)))
        z = np.empty(n, dtype=np.complex128)
        z[:-1] = a_prefix + 1j * sign * b_prefix
        z[-1] = np.cos(theta_n) + 1j * np.sin(theta_n)
        return z

    candidates = [0.0, x_max]
    best_gain = score(0.0)
    best_x = 0.0

    grid_size = NUMERIC_GRID_SIZE
    xs = np.linspace(0.0, x_max, grid_size)
    residuals = residual_grid(xs)

    for i in range(grid_size - 1):
        x_left = float(xs[i])
        x_right = float(xs[i + 1])
        r_left = float(residuals[i])
        r_right = float(residuals[i + 1])

        if not np.isfinite(r_left) or not np.isfinite(r_right):
            continue
        if abs(r_left) <= NUMERIC_X_TOL:
            candidates.append(x_left)
        if r_left * r_right < 0.0:
            candidates.append(brentq(
                residual_scalar,
                x_left,
                x_right,
                xtol=NUMERIC_X_TOL,
                maxiter=NUMERIC_MAX_ITER,
            ))

    if np.isfinite(residuals[-1]) and abs(float(residuals[-1])) <= NUMERIC_X_TOL:
        candidates.append(float(xs[-1]))

    for x in candidates:
        if not 0.0 <= x <= x_max + NUMERIC_X_TOL:
            continue
        g = score(x)
        if g > best_gain:
            best_gain = g
            best_x = x

    best_z = construct_z(best_x)
    best_z = best_z / np.abs(best_z)
    return (U * np.conjugate(best_z)) @ Vh

########################################Running functions

def fmt16(value):
    value = float(value)
    if abs(value) < TOL:
        value = 0.0
    return f"{value:.15e}"


def fmt_ms(value):
    return f"{float(value):.3f}"


def print_polynomial(U, s, Vh, args):
    phase = svd_phase(U, Vh)
    precision = normalize_precision(args.nearest_su_precision)
    if np.min(s) < singular_value_tol(s):
        print("Polynomial: skipped because the matrix is singular; nearest_su defaults to naive_2.")
        return
    if abs(np.imag(phase)) < PHASE_CANDIDATE_TOL:
        if np.real(phase) > 0.0:
            print("Polynomial: skipped because the determinant phase is positive real; nearest_su defaults to naive_2.")
            return
        else:
            print("Polynomial: determinant phase is negative real; nearest_su uses the analytical polynomial branch.")

    coeffs, _, _, _ = build_univariate_polynomial(s, phase, precision)
    print("Polynomial variable: b_n")
    print(f"Polynomial degree: {len(coeffs) - 1}")
    print("Coefficients are in ascending order, i.e. c[k] multiplies b_n^k:")
    for k, c in enumerate(coeffs):
        rendered = mp.nstr(c, 24) if precision == "mp" else f"{float(c):.17g}"
        print(f"  c[{k}] = {rendered}")


def run_trials(args):
    if args.n < 2:
        raise ValueError("experiments require --n >= 2")

    args.nearest_su_precision = normalize_precision(args.nearest_su_precision)
    rng = np.random.default_rng(args.seed)
    methods = [
        ("naive_1", naive_1, "svd"),
        ("naive_2", naive_2, "svd"),
    ]
    if args.n > 2 and args.n <= 7:
        methods.append((
            f"nearest_su_{args.nearest_su_precision}",
            lambda U, s, Vh: nearest_su(U, s, Vh, args.nearest_su_precision, args.mp_dps),
            "svd",
        ))
    elif args.n > 7:
        print(f"Skipping nearest_su_{args.nearest_su_precision}: exact polynomial method is disabled for n > 7.")
    if args.n == 2:
        methods += [
            ("n2_linear_algebra", su2_linear_algebra, "svd"),
            ("n2_algebraic", su2_algebraic, "matrix"),
        ]
    methods.append(("numerical_optimization", numerical_optimization, "svd"))

    nearest_name = f"nearest_su_{args.nearest_su_precision}"
    has_nearest = any(name == nearest_name for name, _, _ in methods)
    results = {name: {"loss": [], "time": [], "valid": 0, "errors": 0, "best": 0} for name, _, _ in methods}
    nearest_count = 0

    for trial in range(args.trials):
        M = make_test_matrix(args, rng)
        U, s, Vh = np.linalg.svd(M, full_matrices=True)

        if args.show_polynomial and trial == 0:
            print_polynomial(U, s, Vh, args)
            if args.polynomial_only:
                return None

        print(f"\nTrial {trial + 1}/{args.trials}")
        trial_losses = {}
        for name, method, input_kind in methods:
            start = time.perf_counter()
            try:
                if input_kind == "matrix":
                    Q = method(M)
                else:
                    if args.n == 2:
                        U_method, s_method, Vh_method = np.linalg.svd(M, full_matrices=True)
                        Q = method(U_method, s_method, Vh_method)
                    else:
                        Q = method(U, s, Vh)
                elapsed = time.perf_counter() - start
                loss = float(np.linalg.norm(M - Q, ord="fro"))
                valid = verify_su(Q)
                results[name]["time"].append(elapsed)
                results[name]["loss"].append(loss)
                results[name]["valid"] += int(valid)
                trial_losses[name] = loss
                print(f"  {name:<24} loss={fmt16(loss)} time_ms={fmt_ms(elapsed * 1000)} valid={valid}")
            except Exception:
                elapsed = time.perf_counter() - start
                results[name]["time"].append(elapsed)
                results[name]["errors"] += 1
                print(f"  {name:<24} loss=n/a time_ms={fmt_ms(elapsed * 1000)} valid=False")

        if trial_losses:
            best_loss = min(trial_losses.values())
            best_methods = [
                name
                for name, loss in trial_losses.items()
                if loss <= best_loss + args.nearest_eps
            ]
            for name in best_methods:
                results[name]["best"] += 1
            print(f"  best_methods={', '.join(best_methods)}")

        if has_nearest:
            if nearest_name in trial_losses:
                is_nearest = trial_losses[nearest_name] <= best_loss + args.nearest_eps
                nearest_count += int(is_nearest)
            else:
                is_nearest = False
            print(f"  nearest_su_is_best={is_nearest}")

    if has_nearest:
        results[nearest_name]["nearest_count"] = nearest_count
    return results


def print_summary(results, trials, timing_note):
    if results is None or not results:
        return

    rows = []
    for name, data in results.items():
        losses = np.array(data["loss"], dtype=float)
        times = np.array(data["time"], dtype=float) * 1000
        rows.append((
            name,
            fmt16(np.mean(losses)) if len(losses) else "n/a",
            fmt16(np.median(losses)) if len(losses) else "n/a",
            fmt_ms(np.mean(times)) if len(times) else "n/a",
            fmt_ms(np.median(times)) if len(times) else "n/a",
            f"{data['valid']}/{trials}",
            f"{data['best']}/{trials}",
        ))

    headers = ("method", "mean_frob", "median_frob", "mean_ms", "median_ms", "valid/trials", "best/trials")
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    alignments = ("<", ">", ">", ">", ">", ">", ">")

    def render(values):
        return "  ".join(
            f"{value:{align}{width}}"
            for value, align, width in zip(values, alignments, widths)
        )

    print("\nSummary")
    print(render(headers))
    print("-" * len(render(headers)))
    for row in rows:
        print(render(row))

    for name, data in results.items():
        if "nearest_count" in data:
            print(f"\n{name} best in {data['nearest_count']} / {trials} trials")

    print(f"\nTiming note: {timing_note}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3, help="matrix dimension")
    parser.add_argument("--trials", type=int, default=20, help="number of random trials")
    parser.add_argument("--seed", type=int, default=1738, help="random seed")
    parser.add_argument("--real_determinant", action="store_true", help="force generated matrices to have real determinant")
    parser.add_argument("--real-determinant", dest="real_determinant", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--singular", action="store_true", help="force one or more singular values to zero")
    parser.add_argument("--zero_singular_values", type=int, default=0, help="number of zero singular values when singular")
    parser.add_argument("--zero-singular-values", dest="zero_singular_values", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--repeated_singular_values", action="store_true", help="force a repeated singular value")
    parser.add_argument("--repeated-singular-values", dest="repeated_singular_values", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--nearest_su_precision", dest="nearest_su_precision", choices=("np", "mp"), default="np", help="precision used by nearest_su")
    parser.add_argument("--nearest-su-precision", dest="nearest_su_precision", choices=("np", "mp"), help=argparse.SUPPRESS)
    parser.add_argument("--mp_dps", type=int, default=MP_DPS, help="mpmath decimal digits")
    parser.add_argument("--mp-dps", dest="mp_dps", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--nearest_eps", type=float, default=NEAREST_EPS, help="epsilon for nearest_su best-method check")
    parser.add_argument("--show_polynomial", action="store_true", help="print the polynomial for the first trial")
    parser.add_argument("--show-polynomial", dest="show_polynomial", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--polynomial_only", action="store_true", help="print the polynomial and skip method evaluation")
    parser.add_argument("--polynomial-only", dest="polynomial_only", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.polynomial_only:
        args.show_polynomial = True
    args.nearest_su_precision = normalize_precision(args.nearest_su_precision)
    if args.n < 2:
        raise ValueError("--n must be at least 2")
    if args.nearest_su_precision == "mp" and mp is None:
        print("Warning: mpmath is not installed in this Python environment; nearest_su_mp will not run.")
    timing_note = (
        "n=2 timings include per-method SVD for SVD-based methods; matrix-direct methods do not include SVD."
        if args.n == 2
        else "Timings exclude SVD and use the shared precomputed decomposition."
    )
    print_summary(run_trials(args), args.trials, timing_note)


if __name__ == "__main__":
    main()
