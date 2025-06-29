import numpy as np
import pytest # Using pytest for assertions and test running

# Adjust import paths if necessary, depending on how tests are run (e.g., from root)
# Assuming tests might be run from the project root directory.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from robot_env import TwoLinkArm
from main import arm_cost_function, TARGET_EE_POS, LINK_LENGTHS, DT, HORIZON, R_CONTROL_COST, Q_EE_FINAL_COST, Q_STATE_FINAL_COST
from ddp import iLQRSolver


# Helper function for numerical gradient (from scratch, or could use a library)
def numerical_gradient(f, x, u, k, is_final, arm_instance, variable_to_diff, epsilon=1e-6):
    """Computes numerical gradient of scalar cost function f wrt variable_to_diff ('x' or 'u')."""
    grad = np.zeros_like(variable_to_diff)

    for i in range(len(variable_to_diff)):
        x_plus = np.copy(x)
        u_plus = np.copy(u)
        x_minus = np.copy(x)
        u_minus = np.copy(u)

        if variable_to_diff is x:
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
        elif variable_to_diff is u:
            u_plus[i] += epsilon
            u_minus[i] -= epsilon
        else: # Should not happen
            raise ValueError("variable_to_diff must be x or u")

        cost_plus, _, _, _, _, _ = arm_cost_function(arm_instance, x_plus, u_plus, k, is_final, False)
        cost_minus, _, _, _, _, _ = arm_cost_function(arm_instance, x_minus, u_minus, k, is_final, False)

        grad[i] = (cost_plus - cost_minus) / (2 * epsilon)
    return grad

# Helper function for numerical Hessian (more involved)
def numerical_hessian_xx(f, x, u, k, is_final, arm_instance, epsilon=1e-5):
    """Computes numerical Hessian of scalar cost function f wrt state x."""
    n = len(x)
    hess = np.zeros((n, n))
    # Using central differences for gradient, then for Hessian elements
    for i in range(n):
        for j in range(n):
            # Perturb x_i and x_j
            x_pp = np.copy(x); x_pp[i] += epsilon; x_pp[j] += epsilon
            x_pm = np.copy(x); x_pm[i] += epsilon; x_pm[j] -= epsilon
            x_mp = np.copy(x); x_mp[i] -= epsilon; x_mp[j] += epsilon
            x_mm = np.copy(x); x_mm[i] -= epsilon; x_mm[j] -= epsilon

            cost_pp,_,_,_,_,_ = arm_cost_function(arm_instance, x_pp, u, k, is_final, False)
            cost_pm,_,_,_,_,_ = arm_cost_function(arm_instance, x_pm, u, k, is_final, False)
            cost_mp,_,_,_,_,_ = arm_cost_function(arm_instance, x_mp, u, k, is_final, False)
            cost_mm,_,_,_,_,_ = arm_cost_function(arm_instance, x_mm, u, k, is_final, False)

            hess[i, j] = (cost_pp - cost_pm - cost_mp + cost_mm) / (4 * epsilon**2)
    return hess

# Similar functions would be needed for L_uu and L_ux, or a more general Hessian function.

def numerical_hessian_uu(f, x, u, k, is_final, arm_instance, epsilon=1e-5):
    """Computes numerical Hessian of scalar cost function f wrt control u."""
    m = len(u)
    hess = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            u_pp = np.copy(u); u_pp[i] += epsilon; u_pp[j] += epsilon
            u_pm = np.copy(u); u_pm[i] += epsilon; u_pm[j] -= epsilon
            u_mp = np.copy(u); u_mp[i] -= epsilon; u_mp[j] += epsilon
            u_mm = np.copy(u); u_mm[i] -= epsilon; u_mm[j] -= epsilon

            cost_pp,_,_,_,_,_ = arm_cost_function(arm_instance, x, u_pp, k, is_final, False)
            cost_pm,_,_,_,_,_ = arm_cost_function(arm_instance, x, u_pm, k, is_final, False)
            cost_mp,_,_,_,_,_ = arm_cost_function(arm_instance, x, u_mp, k, is_final, False)
            cost_mm,_,_,_,_,_ = arm_cost_function(arm_instance, x, u_mm, k, is_final, False)

            hess[i, j] = (cost_pp - cost_pm - cost_mp + cost_mm) / (4 * epsilon**2)
    return hess

def numerical_hessian_ux(f, x, u, k, is_final, arm_instance, epsilon=1e-5):
    """Computes numerical cross-Hessian of scalar cost function f wrt u then x (L_ux)."""
    m = len(u)
    n = len(x)
    hess = np.zeros((m, n)) # Shape (control_dim, state_dim)
    for i in range(m): # Iterate over control dimensions
        for j in range(n): # Iterate over state dimensions
            # Perturb u_i and x_j
            u_p_x_p = np.copy(u); u_p_x_p[i] += epsilon; x_p_u_p = np.copy(x); x_p_u_p[j] += epsilon
            u_p_x_m = np.copy(u); u_p_x_m[i] += epsilon; x_m_u_p = np.copy(x); x_m_u_p[j] -= epsilon
            u_m_x_p = np.copy(u); u_m_x_p[i] -= epsilon; x_p_u_m = np.copy(x); x_p_u_m[j] += epsilon
            u_m_x_m = np.copy(u); u_m_x_m[i] -= epsilon; x_m_u_m = np.copy(x); x_m_u_m[j] -= epsilon

            cost_pp,_,_,_,_,_ = arm_cost_function(arm_instance, x_p_u_p, u_p_x_p, k, is_final, False)
            cost_pm,_,_,_,_,_ = arm_cost_function(arm_instance, x_m_u_p, u_p_x_m, k, is_final, False)
            cost_mp,_,_,_,_,_ = arm_cost_function(arm_instance, x_p_u_m, u_m_x_p, k, is_final, False)
            cost_mm,_,_,_,_,_ = arm_cost_function(arm_instance, x_m_u_m, u_m_x_m, k, is_final, False)

            hess[i, j] = (cost_pp - cost_pm - cost_mp + cost_mm) / (4 * epsilon**2)
    return hess


# Actual test function for pytest
def test_arm_cost_derivatives():
    arm = TwoLinkArm(link_lengths=LINK_LENGTHS)
    # Sample state and control
    x_sample = np.array([np.pi/4, np.pi/6, 0.1, -0.05], dtype=float)
    u_sample = np.array([0.5, -0.2], dtype=float)
    k_sample = 0 # time step

    # --- Test Running Cost Derivatives ---
    # Analytical
    _, lx_an_run, lu_an_run, lxx_an_run, luu_an_run, lux_an_run = \
        arm_cost_function(arm, x_sample, u_sample, k_sample, is_final=False, compute_derivatives=True)

    # Numerical
    lx_num_run = numerical_gradient(arm_cost_function, x_sample, u_sample, k_sample, False, arm, x_sample)
    lu_num_run = numerical_gradient(arm_cost_function, x_sample, u_sample, k_sample, False, arm, u_sample)
    lxx_num_run = numerical_hessian_xx(arm_cost_function, x_sample, u_sample, k_sample, False, arm)
    luu_num_run = numerical_hessian_uu(arm_cost_function, x_sample, u_sample, k_sample, False, arm)
    lux_num_run = numerical_hessian_ux(arm_cost_function, x_sample, u_sample, k_sample, False, arm)

    np.testing.assert_allclose(lx_an_run, lx_num_run, rtol=1e-3, atol=1e-4, err_msg="Running Lx mismatch")
    np.testing.assert_allclose(lu_an_run, lu_num_run, rtol=1e-3, atol=1e-4, err_msg="Running Lu mismatch")
    np.testing.assert_allclose(lxx_an_run, lxx_num_run, rtol=1e-2, atol=1e-3, err_msg="Running Lxx mismatch") # Hessians are more sensitive
    np.testing.assert_allclose(luu_an_run, luu_num_run, rtol=1e-3, atol=1e-4, err_msg="Running Luu mismatch")
    np.testing.assert_allclose(lux_an_run, lux_num_run, rtol=1e-2, atol=1e-3, err_msg="Running Lux mismatch")

    # --- Test Terminal Cost Derivatives ---
    # Analytical
    _, lx_an_term, lu_an_term, lxx_an_term, luu_an_term, lux_an_term = \
        arm_cost_function(arm, x_sample, u_sample, k_sample, is_final=True, compute_derivatives=True) # u_sample is dummy here

    # Numerical (u_sample is dummy for terminal cost, numerical diff should handle it)
    lx_num_term = numerical_gradient(arm_cost_function, x_sample, u_sample, k_sample, True, arm, x_sample)
    # lu_term should be zero, numerical_gradient might produce small non-zeros if u influences cost (it shouldn't for terminal)
    lu_num_term = numerical_gradient(arm_cost_function, x_sample, u_sample, k_sample, True, arm, u_sample)
    lxx_num_term = numerical_hessian_xx(arm_cost_function, x_sample, u_sample, k_sample, True, arm)
    luu_num_term = numerical_hessian_uu(arm_cost_function, x_sample, u_sample, k_sample, True, arm)
    lux_num_term = numerical_hessian_ux(arm_cost_function, x_sample, u_sample, k_sample, True, arm)

    np.testing.assert_allclose(lx_an_term, lx_num_term, rtol=1e-3, atol=1e-4, err_msg="Terminal Lx mismatch")
    np.testing.assert_allclose(lu_an_term, lu_num_term, rtol=1e-3, atol=1e-5, err_msg="Terminal Lu mismatch") # Should be close to zero
    np.testing.assert_allclose(lxx_an_term, lxx_num_term, rtol=1e-2, atol=1e-3, err_msg="Terminal Lxx mismatch")
    np.testing.assert_allclose(luu_an_term, luu_num_term, rtol=1e-3, atol=1e-5, err_msg="Terminal Luu mismatch") # Should be close to zero
    np.testing.assert_allclose(lux_an_term, lux_num_term, rtol=1e-2, atol=1e-5, err_msg="Terminal Lux mismatch") # Should be close to zero


def test_ilqr_convergence_on_arm():
    """
    Tests if the iLQRSolver converges for a simple reaching task with the TwoLinkArm
    and if the final end-effector position is close to the target.
    """
    # --- Test Setup ---
    # Use constants from main, but can be overridden for faster/simpler test
    test_link_lengths = LINK_LENGTHS
    test_dt = DT
    test_horizon = 20 # Shorter horizon for faster test
    test_initial_angles_deg = [0, 0]
    test_initial_omegas = [0, 0]

    # More reachable target for a short horizon
    test_target_ee_pos = np.array([0.5, 0.5])

    # Cost function weights (can be adjusted for test stability/speed)
    # Using global cost weights from main for consistency, but could be tuned here
    # R_CONTROL_COST, Q_EE_FINAL_COST, Q_STATE_FINAL_COST

    arm = TwoLinkArm(link_lengths=test_link_lengths, initial_angles_rad=np.deg2rad(test_initial_angles_deg))
    arm.dt = test_dt

    initial_state = arm.reset(initial_angles_rad=np.deg2rad(test_initial_angles_deg),
                              initial_omegas_rad_s=test_initial_omegas)

    # Define cost function for the test (can reuse main.arm_cost_function by setting global TARGET_EE_POS if needed)
    # For isolated test, better to define target within test or pass it to cost fn
    # We need to ensure TARGET_EE_POS used by arm_cost_function is our test_target_ee_pos
    # A simple way is to temporarily modify the global, or refactor arm_cost_function

    # Temporary override of global TARGET_EE_POS for this test
    # This is a common pattern but has side effects if tests run in parallel or TARGET_EE_POS is used elsewhere.
    # A cleaner way would be for arm_cost_function to accept target_ee_pos as an argument.
    # For now, we'll use this simple approach for the test.
    original_target_ee_pos = np.copy(TARGET_EE_POS) # Save original
    globals()['TARGET_EE_POS'] = test_target_ee_pos # Override

    test_cost_fn = lambda state_x, control_u, k_step, is_final, compute_derivs: \
                       arm_cost_function(arm, state_x, control_u, k_step, is_final, compute_derivs)

    solver = iLQRSolver(dynamics_fn=arm.dynamics,
                        cost_fn=test_cost_fn,
                        state_dim=arm.state_dim,
                        control_dim=arm.control_dim,
                        horizon=test_horizon)

    U_initial_guess = np.random.randn(test_horizon, arm.control_dim) * 0.001

    # --- Run Solver ---
    # Reduce max_iters for faster test, adjust tolerance
    X_opt, U_opt, final_cost, cost_history = solver.run(
        initial_state, U_initial_guess, max_iters=15, tol=1e-2
    )

    # Restore original TARGET_EE_POS
    globals()['TARGET_EE_POS'] = original_target_ee_pos

    # --- Assertions ---
    assert len(cost_history) > 1, "Solver did not run for at least one iteration."
    initial_cost = cost_history[0]
    assert final_cost < initial_cost, \
        f"Final cost ({final_cost}) not less than initial cost ({initial_cost}). Cost history: {cost_history}"

    final_angles_optimized = X_opt[-1, :2]
    _, _, final_ee_pos_optimized = arm.forward_kinematics(final_angles_optimized)

    # Check if final EE position is reasonably close to the target
    # Tolerance can be adjusted based on problem difficulty and solver settings
    ee_error_final = np.linalg.norm(final_ee_pos_optimized - test_target_ee_pos)
    assert ee_error_final < 0.2, \
        f"Final EE error ({ee_error_final:.4f}) is too large. Final EE: {final_ee_pos_optimized}, Target: {test_target_ee_pos}"

    # Optional: Check if final velocities are small (if Q_STATE_FINAL_COST penalizes them)
    final_velocities = X_opt[-1, 2:]
    assert np.allclose(final_velocities, 0, atol=0.1), \
        f"Final velocities ({final_velocities}) are not close to zero."


# Placeholder for future tests
def test_example_placeholder():
    assert True

# More tests will be added in subsequent steps.
# This file sets up the basic structure and imports.
# print("tests/test_main_components.py created successfully.") # Remove this print

if __name__ == '__main__':
    # Example of how to use numerical gradient (not part of pytest tests directly)
    arm_test = TwoLinkArm(link_lengths=LINK_LENGTHS)
    test_x = np.array([0.1, 0.2, 0.05, 0.05])
    test_u = np.array([0.01, 0.01])

    print("Testing numerical_gradient for lx (running cost):")
    lx_num = numerical_gradient(arm_cost_function, test_x, test_u, 0, False, arm_test, test_x)
    print(f"Numerical lx: {lx_num}")

    _, lx_an, _, _, _, _ = arm_cost_function(arm_test, test_x, test_u, 0, False, True)
    print(f"Analytical lx: {lx_an}")

    print("\\nTesting numerical_gradient for lu (running cost):")
    lu_num = numerical_gradient(arm_cost_function, test_x, test_u, 0, False, arm_test, test_u)
    print(f"Numerical lu: {lu_num}")
    _, _, lu_an, _, _, _ = arm_cost_function(arm_test, test_x, test_u, 0, False, True)
    print(f"Analytical lu: {lu_an}")

    print("\\nTesting numerical_hessian_xx for lxx (running cost):")
    # Note: Analytical lxx for running cost (control only) is zero in current arm_cost_function
    lxx_num_run = numerical_hessian_xx(arm_cost_function, test_x, test_u, 0, False, arm_test)
    print(f"Numerical lxx (running):\\n{lxx_num_run}")
    _, _, _, lxx_an_run, _, _ = arm_cost_function(arm_test, test_x, test_u, 0, False, True)
    print(f"Analytical lxx (running):\\n{lxx_an_run}")

    print("\\nTesting numerical_hessian_xx for lxx (final cost):")
    lxx_num_final = numerical_hessian_xx(arm_cost_function, test_x, test_u, 0, True, arm_test) # k=0, is_final=True for test
    print(f"Numerical lxx (final):\\n{lxx_num_final}")
    _, _, _, lxx_an_final, _, _ = arm_cost_function(arm_test, test_x, test_u, 0, True, True)
    print(f"Analytical lxx (final):\\n{lxx_an_final}")
