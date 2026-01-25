import numpy as np
import matplotlib.pyplot as plt

from robot_env import TwoLinkArm, RobotVisualizer
from ddp import iLQRSolver # Updated import from DDP to iLQRSolver

# === Configuration Constants ===
# --- Robot Arm Parameters ---
LINK_LENGTHS = [1.0, 1.0]  # Lengths of link1 and link2
INITIAL_ANGLES_DEG = [0.0, 0.0] # Initial joint angles [theta1, theta2] in degrees
INITIAL_OMEGAS_RAD_S = [0.0, 0.0] # Initial joint velocities [omega1, omega2] in rad/s
DT = 0.1  # Time step for simulation and DDP (seconds)
HORIZON = 30  # Number of time steps for DDP trajectory (N) # Reduced from 50 for faster testing

# --- Target Definition ---
TARGET_EE_POS = np.array([1.2, 0.8]) # Target end-effector [x,y] position

# --- Cost Function Weights ---
# These weights define the behavior of the DDP optimization.
# Adjust them to prioritize different aspects (e.g., accuracy vs. control effort).
R_CONTROL_COST = 0.01       # Penalty for control effort (sum of u^T R u for each step)
Q_EE_FINAL_COST = 1000.0    # Penalty for end-effector position error at the final time step
Q_STATE_FINAL_COST = 10.0   # Penalty for deviation from a desired final state (e.g., zero velocity)
# Q_RUNNING_STATE_COST = 0.1 # (Optional) Penalty for state deviation during trajectory
# Q_RUNNING_EE_COST = 1.0    # (Optional) Penalty for EE deviation during trajectory


def get_arm_fk_jacobian(arm_instance, joint_angles_rad):
    """
    Computes the Jacobian of the end-effector position with respect to joint angles.
    The Jacobian maps joint velocities to end-effector linear velocities: v_ee = J * omega_joints.
    It's used here to find derivatives of the EE position cost term.

    J = [[∂x_ee/∂theta1, ∂x_ee/∂theta2],
         [∂y_ee/∂theta1, ∂y_ee/∂theta2]]

    Args:
        arm_instance (TwoLinkArm): The robot arm object.
        joint_angles_rad (np.array): Joint angles [theta1, theta2] in radians.

    Returns:
        np.array: The 2x2 Jacobian matrix.
    """
    l1, l2 = arm_instance.link_lengths
    theta1, theta2_rel = joint_angles_rad # theta2_rel is relative to link1

    # Absolute angle of link 2: theta_abs2 = theta1 + theta2_rel
    theta_abs2 = theta1 + theta2_rel

    s1 = np.sin(theta1)
    c1 = np.cos(theta1)
    s_abs2 = np.sin(theta_abs2) # sin(theta1 + theta2_rel)
    c_abs2 = np.cos(theta_abs2) # cos(theta1 + theta2_rel)

    # Partial derivatives of end-effector position (x_ee, y_ee)
    # x_ee = l1*c1 + l2*c_abs2
    # y_ee = l1*s1 + l2*s_abs2

    # ∂x_ee/∂theta1 = -l1*s1 - l2*s_abs2 * (∂theta_abs2/∂theta1) = -l1*s1 - l2*s_abs2 * 1
    # ∂x_ee/∂theta2_rel = -l2*s_abs2 * (∂theta_abs2/∂theta2_rel) = -l2*s_abs2 * 1
    dx_dtheta1 = -l1 * s1 - l2 * s_abs2
    dx_dtheta2_rel = -l2 * s_abs2

    # ∂y_ee/∂theta1 = l1*c1 + l2*c_abs2 * (∂theta_abs2/∂theta1) = l1*c1 + l2*c_abs2 * 1
    # ∂y_ee/∂theta2_rel = l2*c_abs2 * (∂theta_abs2/∂theta2_rel) = l2*c_abs2 * 1
    dy_dtheta1 = l1 * c1 + l2 * c_abs2
    dy_dtheta2_rel = l2 * c_abs2

    jacobian = np.array([
        [dx_dtheta1, dx_dtheta2_rel],
        [dy_dtheta1, dy_dtheta2_rel]
    ])
    return jacobian


def arm_cost_function(arm_instance, state_x, control_u, time_step_k, is_final_step=False, compute_derivatives=False):
    """
    Cost function for the TwoLinkArm tailored for DDP.
    State: [theta1, theta2, omega1, omega2] (angles in rad, velocities in rad/s)
    Control: [alpha1, alpha2] (angular accelerations in rad/s^2)

    Args:
        arm_instance (TwoLinkArm): The robot arm object.
        state_x (np.array): Current state vector.
        control_u (np.array): Current control vector.
        time_step_k (int): Current time step index.
        is_final_step (bool): True if this is the terminal cost calculation.
        compute_derivatives (bool): If True, computes and returns cost derivatives (lx, lu, lxx, luu, lux).

    Returns:
        tuple: (cost, lx, lu, lxx, luu, lux)
               If compute_derivatives is False, derivative terms will be zero arrays.
    """
    # Initialize derivative terms (gradients and Hessians)
    lx = np.zeros_like(state_x)
    lu = np.zeros_like(control_u)
    lxx = np.zeros((len(state_x), len(state_x)))
    luu = np.zeros((len(control_u), len(control_u)))
    lux = np.zeros((len(control_u), len(state_x))) # or lux.T is lxu

    total_cost = 0.0

    # --- Terminal Cost (applied only at the final time step N) ---
    if is_final_step:
        # 1. End-effector position cost: Penalize distance from target EE position
        current_joint_angles = state_x[:2] # Extract [theta1, theta2]
        _, _, current_ee_pos = arm_instance.forward_kinematics(current_joint_angles)
        ee_error = current_ee_pos - TARGET_EE_POS
        cost_ee_final = 0.5 * Q_EE_FINAL_COST * np.dot(ee_error, ee_error)
        total_cost += cost_ee_final

        # 2. Final state cost: Penalize deviation from a desired final state
        #    Example: Encourage zero velocity at the end.
        #    Target final state: current angles, zero velocities.
        target_final_state = np.array([state_x[0], state_x[1], 0.0, 0.0])
        state_error_final = state_x - target_final_state
        cost_state_final = 0.5 * Q_STATE_FINAL_COST * np.dot(state_error_final, state_error_final)
        total_cost += cost_state_final

        if compute_derivatives:
            # Derivatives for end-effector position cost (Q_EE_FINAL_COST)
            # lx_ee = J_fk^T * Q_EE_FINAL_COST * ee_error
            # lxx_ee = J_fk^T * Q_EE_FINAL_COST * J_fk (Gauss-Newton approximation)
            J_fk = get_arm_fk_jacobian(arm_instance, current_joint_angles) # Jacobian (2x2 for 2-link arm)

            # Gradient lx (only for angle components theta1, theta2)
            lx[:2] = Q_EE_FINAL_COST * J_fk.T @ ee_error
            # Hessian lxx (only for angle components)
            lxx[:2, :2] = Q_EE_FINAL_COST * (J_fk.T @ J_fk) # Gauss-Newton approx. for Hessian

            # Derivatives for final state cost (Q_STATE_FINAL_COST)
            lx += Q_STATE_FINAL_COST * state_error_final # Add to lx from EE cost
            lxx += Q_STATE_FINAL_COST * np.eye(len(state_x)) # Add to lxx

            # lu, luu, lux are zero for the terminal cost as control is not applied at the final state.

    # --- Running Cost (applied at each time step k < N) ---
    else:
        # 1. Control effort cost: Penalize large control inputs (accelerations)
        cost_control = 0.5 * R_CONTROL_COST * np.dot(control_u, control_u)
        total_cost += cost_control

        # (Optional) Add running state cost or running EE position cost here if desired
        # Example: Penalize deviation from a reference trajectory or keep EE close to a path.
        # current_joint_angles = state_x[:2]
        # _, _, current_ee_pos = arm_instance.forward_kinematics(current_joint_angles)
        # running_ee_error = current_ee_pos - TARGET_EE_POS # Or some intermediate target
        # cost_running_ee = 0.5 * Q_RUNNING_EE_COST * np.dot(running_ee_error, running_ee_error)
        # total_cost += cost_running_ee

        if compute_derivatives:
            # Derivatives for control cost
            lu = R_CONTROL_COST * control_u
            luu = R_CONTROL_COST * np.eye(len(control_u))

            # If optional running costs were added, their derivatives would be computed here too.
            # For Q_RUNNING_EE_COST:
            # J_fk = get_arm_fk_jacobian(arm_instance, current_joint_angles)
            # lx_running_ee = Q_RUNNING_EE_COST * J_fk.T @ running_ee_error
            # lx[:2] += lx_running_ee
            # lxx_running_ee = Q_RUNNING_EE_COST * (J_fk.T @ J_fk)
            # lxx[:2,:2] += lxx_running_ee

    return total_cost, lx, lu, lxx, luu, lux


def main():
    """Main function to set up and run the DDP optimization for the 2-link arm."""

    # --- 1. Initialize Robot Arm Environment ---
    initial_angles_rad = np.deg2rad(INITIAL_ANGLES_DEG)
    arm = TwoLinkArm(link_lengths=LINK_LENGTHS, initial_angles_rad=initial_angles_rad)
    arm.dt = DT # Ensure arm's dt matches DDP horizon dt for consistency

    # Initial state vector [theta1, theta2, omega1, omega2]
    initial_state = arm.reset(initial_angles_rad=initial_angles_rad, initial_omegas_rad_s=INITIAL_OMEGAS_RAD_S)


    # --- 2. Define Dynamics Function for DDP ---
    # The DDP solver expects a function f(state, control) -> next_state.
    # The TwoLinkArm.dynamics method already fits this signature.
    dynamics_fn_for_ddp = arm.dynamics


    # --- 3. Define Cost Function for DDP ---
    # The DDP solver expects l(state, control, k, is_final, get_derivatives) -> cost, lx, lu, lxx, luu, lux.
    # We use a lambda to partially apply the `arm` instance to our `arm_cost_function`.
    cost_fn_for_ddp = lambda state_x, control_u, k_step, is_final, compute_derivs: \
                       arm_cost_function(arm, state_x, control_u, k_step, is_final, compute_derivs)


    # --- 4. Initialize iLQR Solver ---
    # Option A: Using existing numerical differentiation within iLQRSolver
    ilqr_solver = iLQRSolver(dynamics_fn=dynamics_fn_for_ddp,
                             cost_fn=cost_fn_for_ddp, # This cost_fn must provide its own derivatives
                             state_dim=arm.state_dim,
                             control_dim=arm.control_dim,
                             horizon=HORIZON)

    # --- 5. Initial Control Guess for iLQR ---
    # A common starting point is zero controls.
    # Small random controls can sometimes help break symmetries or explore if zero gets stuck.
    # U_initial_guess = np.zeros((HORIZON, arm.control_dim))
    U_initial_guess = np.random.randn(HORIZON, arm.control_dim) * 0.001 # Small random accelerations


    # --- 6. Run iLQR Optimization --- # Changed DDP to iLQR
    print("Running iLQR optimization for the 2-Link Arm...") # Changed DDP to iLQR
    # Note: max_iters might need adjustment based on problem complexity and desired convergence.
    # Numerical differentiation for dynamics (default in this iLQR impl) can be slow.
    # Analytical dynamics derivatives (fx, fu) would significantly speed this up.
    X_opt, U_opt, final_cost, cost_history = ilqr_solver.run( # Changed ddp_solver to ilqr_solver
        initial_state,
        U_initial_guess,
        max_iters=25, # Adjusted for potentially faster run in this environment
        tol=1e-4
    )

    print("\n--- iLQR Optimization Results ---") # Changed DDP to iLQR
    print(f"Final Optimized Cost: {final_cost:.4f}")
    print(f"Initial state: {X_opt[0]}")
    print(f"Final state (angles, omegas): {X_opt[-1]}")

    final_angles_optimized = X_opt[-1, :2]
    _, _, final_ee_pos_optimized = arm.forward_kinematics(final_angles_optimized)
    print(f"Final End-Effector Position: {final_ee_pos_optimized}")
    print(f"Target End-Effector Position: {TARGET_EE_POS}")
    ee_error_final = np.linalg.norm(final_ee_pos_optimized - TARGET_EE_POS)
    print(f"Final EE Error: {ee_error_final:.4f}")


    # 7. Visualize the Optimized Trajectory
    print("\nVisualizing optimized trajectory...")
    vis = RobotVisualizer(arm, target_pos=TARGET_EE_POS)

    # Show initial position
    vis.update_plot(X_opt[0, :2]) # Plot initial angles
    if plt.get_backend() != 'agg': # pragma: no cover
        plt.pause(1)

    # Animate the optimized trajectory
    # The animation function in RobotVisualizer expects a list of states
    # where each state is [theta1, theta2, omega1, omega2]
    robot_animation_fig = vis.fig # Get the figure from the visualizer
    animation = vis.animate_trajectory(X_opt, interval=int(DT * 1000))

    # --- Additional Plots ---
    # Plot 1: Cost history
    fig_cost, ax_cost = plt.subplots()
    ax_cost.plot(cost_history)
    ax_cost.set_xlabel('iLQR Iteration') # Changed DDP to iLQR
    ax_cost.set_ylabel('Total Cost')
    ax_cost.set_title('Cost Function Evolution over iLQR Iterations') # Changed DDP to iLQR
    ax_cost.grid(True)
    if plt.get_backend() == 'agg': # pragma: no cover
        fig_cost.savefig("ilqr_cost_history.png") # Changed DDP to iLQR
        print("Cost history plot saved to ilqr_cost_history.png")


    # Plot 2: Control inputs
    time_u = np.arange(HORIZON) * DT
    fig_ctrl, axs_ctrl = plt.subplots(arm.control_dim, 1, sharex=True, figsize=(8, 6))
    if arm.control_dim == 1: axs_ctrl = [axs_ctrl] # Make iterable if single control
    for i in range(arm.control_dim):
        axs_ctrl[i].plot(time_u, U_opt[:, i])
        axs_ctrl[i].set_ylabel(f'Control u{i+1} (accel)')
        axs_ctrl[i].grid(True)
    axs_ctrl[-1].set_xlabel('Time (s)')
    fig_ctrl.suptitle('iLQR Optimized Control Inputs (Joint Accelerations)') # Changed
    if plt.get_backend() == 'agg': # pragma: no cover
        fig_ctrl.savefig("ilqr_optimized_controls.png") # Changed
        print("Optimized controls plot saved to ilqr_optimized_controls.png")

    # Plot 3: State trajectory (angles and velocities)
    time_x = np.arange(HORIZON + 1) * DT
    fig_state, axs_state = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    # Plot joint angles
    axs_state[0].plot(time_x, np.rad2deg(X_opt[:, 0]), label=r'$\theta_1$ (deg)')
    axs_state[0].plot(time_x, np.rad2deg(X_opt[:, 1]), label=r'$\theta_2$ (deg)')
    axs_state[0].set_ylabel('Joint Angles')
    axs_state[0].legend()
    axs_state[0].grid(True)
    # Plot joint velocities
    axs_state[1].plot(time_x, X_opt[:, 2], label=r'$\omega_1$ (rad/s)')
    axs_state[1].plot(time_x, X_opt[:, 3], label=r'$\omega_2$ (rad/s)')
    axs_state[1].set_ylabel('Joint Velocities')
    axs_state[1].set_xlabel('Time (s)')
    axs_state[1].legend()
    axs_state[1].grid(True)
    fig_state.suptitle('iLQR Optimized State Trajectory') # Changed
    if plt.get_backend() == 'agg': # pragma: no cover
        fig_state.savefig("ilqr_optimized_states.png") # Changed
        print("Optimized states plot saved to ilqr_optimized_states.png")

    # Plot 4: End-effector path (iLQR vs Naive) # Changed
    ee_path_ilqr = np.array([arm.forward_kinematics(X_opt[i, :2])[2] for i in range(HORIZON + 1)]) # Renamed ddp to ilqr

    # --- Generate and Plot Naive Trajectory for Comparison ---
    # Define a target joint angle configuration for the naive trajectory.
    # This is a heuristic guess; a proper Inverse Kinematics (IK) solver
    # would be better for finding angles that precisely match TARGET_EE_POS.
    # The current target aims for an EE position somewhat near the iLQR target for visual comparison. # Changed DDP to iLQR
    TARGET_ANGLES_NAIVE_DEG = [25, 20] # Example: [theta1_deg, theta2_deg]
    target_angles_naive_rad = np.deg2rad(TARGET_ANGLES_NAIVE_DEG)

    X_naive, U_naive, cost_naive = generate_naive_trajectory(
        arm, initial_state, target_angles_naive_rad,
        HORIZON, DT, cost_fn_for_ddp # Pass the iLQR cost function for fair comparison
    )
    ee_path_naive = np.array([arm.forward_kinematics(X_naive[i, :2])[2] for i in range(HORIZON + 1)])

    print(f"\n--- Cost Comparison ---")
    print(f"iLQR Optimized Trajectory Cost: {final_cost:.2f}") # Changed DDP to iLQR
    print(f"Naive Trajectory Cost: {cost_naive:.2f}")


    # Update End-Effector Path plot to include naive trajectory
    fig_path, ax_path = plt.subplots(figsize=(8,8)) # Slightly larger for better legend display
    ax_path.plot(ee_path_ilqr[:, 0], ee_path_ilqr[:, 1], 'b-', lw=2.5, label=f'iLQR Path (Cost: {final_cost:.2f})')
    ax_path.plot(ee_path_naive[:, 0], ee_path_naive[:, 1], 'm--', lw=2, label=f'Naive Path (Cost: {cost_naive:.2f})')

    # Mark start and target points
    ax_path.plot(ee_path_ilqr[0, 0], ee_path_ilqr[0, 1], 'go', ms=10, label='Start EE') # Changed ee_path_ddp to ee_path_ilqr
    ax_path.plot(TARGET_EE_POS[0], TARGET_EE_POS[1], 'rx', ms=10, mew=2, label='Target EE') # Red 'x'
    ax_path.set_xlabel('X Position (m)')
    ax_path.set_ylabel('Y Position (m)')
    ax_path.set_title('End-Effector Path')
    ax_path.legend()
    ax_path.axis('equal')
    ax_path.grid(True)
    if plt.get_backend() == 'agg': # pragma: no cover
        fig_path.savefig("ilqr_ee_path.png") # Changed
        print("End-effector path plot saved to ilqr_ee_path.png")


    if plt.get_backend() != 'agg': # pragma: no cover
        print("Displaying animation and plots. Close the plot windows to exit.")
        plt.show() # Show all figures
    else:
        print("Running in 'agg' backend. Animation will not be shown directly. Plots saved to files.")
        # Ensure robot animation figure is also saved if not shown
        try:
            robot_animation_fig.savefig("ilqr_robot_animation_final_frame.png") # Changed
            print("Robot animation final frame saved to ilqr_robot_animation_final_frame.png")
            # Saving full animation as GIF needs FuncAnimation object and writer
            # animation.save('arm_ilqr_trajectory.gif', writer='imagemagick', fps=1.0/DT) # Changed
            # print("Animation saved to arm_ilqr_trajectory.gif")
        except Exception as e:
            print(f"Could not save robot animation figure/gif: {e}")


    print("Main script finished.")


def generate_naive_trajectory(arm, initial_state_arm, target_angles_naive_rad, horizon, dt, cost_fn_for_comparison):
    """
    Generates a 'naive' trajectory for comparison with the DDP optimized one.
    This trajectory is created by:
    1. Defining a target joint angle configuration (`target_angles_naive_rad`).
    2. Calculating constant joint velocities required to linearly interpolate from the
       initial joint angles to these target angles over the specified horizon.
    3. Simulating the arm's movement using a simple P-controller on velocity error
       to generate joint accelerations (controls). The goal is to track the
       calculated constant joint velocities.

    This approach is 'naive' because it doesn't optimize a cost function considering
    dynamics and control effort throughout the path, unlike DDP.

    Args:
        arm (TwoLinkArm): The robot arm instance.
        initial_state_arm (np.array): Initial state of the arm [theta1, theta2, omega1, omega2].
        target_angles_naive_rad (np.array): Target joint angles [theta1, theta2] for the naive trajectory.
        horizon (int): Number of time steps for the trajectory.
        dt (float): Time step duration.
        cost_fn_for_comparison (callable): The same cost function used by DDP, for fair cost evaluation.
                                         Expected signature: (state, control, k, is_final, get_derivs) -> cost, ...

    Returns:
        tuple: (X_naive_sim, U_naive_sim, naive_total_cost)
            X_naive_sim (np.array): State trajectory of the naive approach (horizon+1, state_dim).
            U_naive_sim (np.array): Control trajectory of the naive approach (horizon, control_dim).
            naive_total_cost (float): Total cost of the naive trajectory.
    """
    print("\n--- Generating Naive Trajectory (Linear Angle Interpolation with P-Control on Velocity) ---")
    initial_angles_rad = initial_state_arm[:2] # Current joint angles
    initial_omegas_rad_s = initial_state_arm[2:] # Current joint velocities

    # Initialize state and control trajectories for the naive approach
    X_naive_sim = np.zeros((horizon + 1, arm.state_dim))
    U_naive_sim = np.zeros((horizon, arm.control_dim))

    current_state_sim = np.copy(initial_state_arm)
    X_naive_sim[0,:] = current_state_sim

    # 1. Calculate target constant joint velocities to reach `target_angles_naive_rad`
    #    from `initial_angles_rad` in `horizon * dt` seconds.
    total_time = horizon * dt
    if total_time <= 1e-6: # Avoid division by zero if horizon or dt is too small
        target_joint_velocities = np.zeros_like(initial_angles_rad)
    else:
        target_joint_velocities = (target_angles_naive_rad - initial_angles_rad) / total_time

    # 2. Simulate using a P-controller on velocity error to generate accelerations
    #    The controller tries to make the arm's joint velocities match `target_joint_velocities`.
    #    A more sophisticated naive approach might use a PD controller on a joint angle trajectory.
    Kp_vel_tracking = 2.5  # Proportional gain for the velocity tracking P-controller
                           # This gain might need tuning for different dynamics or dt.
    MAX_NAIVE_ACCEL = np.pi # Max acceleration for naive controller (to keep it somewhat bounded)


    for k in range(horizon):
        # Calculate velocity error: e_vel = desired_velocity - current_velocity
        current_joint_velocities = current_state_sim[2:]
        velocity_error = target_joint_velocities - current_joint_velocities

        # Control (acceleration) = Kp * velocity_error
        # This attempts to drive the current velocities towards the target constant velocities.
        calculated_accel = Kp_vel_tracking * velocity_error

        # Clip accelerations to a reasonable maximum (optional, but good for stability)
        applied_accel = np.clip(calculated_accel, -MAX_NAIVE_ACCEL, MAX_NAIVE_ACCEL)
        U_naive_sim[k,:] = applied_accel

        # Simulate one step forward using the arm's dynamics
        current_state_sim = arm.dynamics(current_state_sim, applied_accel)
        X_naive_sim[k+1,:] = current_state_sim

    # 3. Calculate the total cost of this naive trajectory using the DDP's cost function
    naive_total_cost = 0.0
    for k in range(horizon):
        cost_k, _, _, _, _, _ = cost_fn_for_comparison(
            X_naive_sim[k,:], U_naive_sim[k,:], k, is_final_step=False, compute_derivatives=False
        )
        naive_total_cost += cost_k

    # Add terminal cost for the final state reached by the naive trajectory
    # Pass a dummy zero control as it's not used in terminal cost calculation.
    cost_final, _, _, _, _, _ = cost_fn_for_comparison(
        X_naive_sim[horizon,:], np.zeros(arm.control_dim), horizon, is_final_step=True, compute_derivatives=False
    )
    naive_total_cost += cost_final

    # --- Output Naive Trajectory Info ---
    print(f"Naive Trajectory - Target Joint Angles (deg): {np.rad2deg(target_angles_naive_rad)}")
    print(f"Naive Trajectory - Calculated Constant Velocities (rad/s): {target_joint_velocities}")
    final_ee_pos_naive = arm.forward_kinematics(X_naive_sim[horizon, :2])[2] # Get EE pos from final angles
    final_ee_error_naive = np.linalg.norm(final_ee_pos_naive - TARGET_EE_POS)
    print(f"Naive Trajectory - Final Simulated EE Position: {final_ee_pos_naive}, Error from Target: {final_ee_error_naive:.4f}")
    print(f"Naive Trajectory - Final State (angles_rad, omegas_rad/s): {X_naive_sim[horizon,:]}")
    # Note: The naive trajectory does not explicitly try to stop (zero velocity) at the end unless
    # target_angles_naive_rad implied that via target_joint_velocities becoming zero.
    # The cost function's Q_STATE_FINAL_COST will penalize non-zero final velocities.

    return X_naive_sim, U_naive_sim, naive_total_cost


if __name__ == '__main__':
    main()
