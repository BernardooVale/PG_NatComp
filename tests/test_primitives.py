import numpy as np
import pytest
from gp.primitives import PRIMITIVES, protected_div, protected_log, protected_sqrt, exp


def test_protected_div():
    a = np.array([10.0, 5.0, 0.0, -4.0])
    b = np.array([2.0, 1e-7, 0.0, -2.0])
    res = protected_div(a, b)
    assert res[0] == 5.0
    assert res[1] == 1.0  # protected against |b| < 1e-6
    assert res[2] == 1.0  # protected against zero division
    assert res[3] == 2.0


def test_protected_log_and_sqrt():
    x = np.array([-4.0, 0.0, 9.0])
    sq = protected_sqrt(x)
    assert np.allclose(sq, [2.0, 0.0, 3.0])

    lg = protected_log(x)
    assert np.all(np.isfinite(lg))


def test_exp_clipping():
    x = np.array([1000.0, -1000.0, 0.0])
    res = exp(x)
    assert np.all(np.isfinite(res))
    assert res[2] == 1.0
