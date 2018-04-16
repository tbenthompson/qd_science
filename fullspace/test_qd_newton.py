import numpy as np

import cppimport.import_hook
import qd_newton

def test_newton():
    result = qd_newton.newton(
        lambda x: np.sqrt(x) - 7,
        lambda x: 0.5 / np.sqrt(x),
        1.0,
        1e-15,
        20
    )
    assert(result[1])
    np.testing.assert_almost_equal(np.sqrt(result[0]), 7.0, 15)

def test_rs_eqtns():
    a = 0.015
    V0 = 1e-6
    sigma_n = 1.0
    state = 0.8
    V, f, fp = np.load('qd_rs_jacobian.npy')
    np.testing.assert_almost_equal(
        np.array([
            qd_newton.F(V[i], sigma_n, state, a, V0)
            for i in range(V.shape[0])
        ]),
        f
    )
    np.testing.assert_almost_equal(
        np.array([
            qd_newton.dFdV(V[i], sigma_n, state, a, V0)
            for i in range(V.shape[0])
        ]),
        fp
    )
    fp_midpt = np.array([
        qd_newton.dFdV((V[i] + V[i + 1]) / 2.0, sigma_n, state, a, V0)
        for i in range(V.shape[0] - 1)
    ])
    fp_finite_diff = (f[1:] - f[:-1]) / (V[1:] - V[:-1])
    fp_err = np.abs((fp_midpt - fp_finite_diff) / fp_midpt)
    np.testing.assert_almost_equal(fp_err, 0.0, 2)

def test_newton_rs():
    correct = 1.6297892292e-10
    result = qd_newton.newton_rs(
        36834658.0767, 4500000.0, 50000000.0, 0.86752150661, 0.015, 1e-06,
        0.0, 1e-16, 30
    )
    assert(result[1])
    np.testing.assert_almost_equal(result[0], correct)


def test_rate_state_solver():
    (
        tri_normals, traction, state, fV2, eta, a, V0,
        additional_normal_stress, V_new_correct
    ) = np.load('qd_newton_test.npy')
    a = np.ones_like(state) * a
    fV2[:] = 0.0
    qd_newton.rate_state_solver(
        tri_normals, traction, state, fV2, a, eta, V0,
        additional_normal_stress, 1e-12, 50
    )
    np.testing.assert_almost_equal(fV2, V_new_correct)

def test_rate_state_solver_zero_tau():
    (
        tri_normals, traction, state, V, eta, a, V0,
        additional_normal_stress, V_new_correct
    ) = np.load('qd_newton_test.npy')
    V[:] = 0.0
    qd_newton.rate_state_solver(
        tri_normals, 0 * traction, state, V, eta, a, V0,
        additional_normal_stress, 1e-12, 50
    )
    assert(not np.any(np.isnan(V)))
