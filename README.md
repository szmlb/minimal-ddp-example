# Differential Dynamic Programming (DDP) for 2-Link Arm Motion Planning

This project demonstrates the application of the Differential Dynamic Programming (DDP) algorithm, specifically an Iterative Linear Quadratic Regulator (iLQR) variant, for optimizing the motion of a simple 2-link planar robotic arm. The goal is to drive the arm's end-effector to a target position while minimizing control effort and satisfying other objectives defined in a cost function.

## Project Overview

The project includes:
1.  **A 2-Link Robotic Arm Simulation (`src/robot_env.py`):**
    *   Defines the arm's kinematics (`forward_kinematics`) and dynamics (`dynamics`).
    *   The state is `[theta1, theta2, omega1, omega2]` (joint angles and velocities).
    *   The control input is `[alpha1, alpha2]` (joint angular accelerations).
    *   Includes a `RobotVisualizer` class for animating the arm's movement using Matplotlib.
2.  **DDP (iLQR) Implementation (`src/ddp.py`):**
    *   A general DDP/iLQR solver that iteratively optimizes a trajectory.
    *   Uses numerical differentiation for system dynamics Jacobians (`fx`, `fu`).
    *   Expects the cost function to provide analytical first and second derivatives (`lx, lu, lxx, luu, lux`).
    *   Features a backward pass to compute control updates (feedforward and feedback gains) and a forward pass with line search to ensure cost reduction.
3.  **Main Application Script (`src/main.py`):**
    *   Sets up the 2-link arm environment and the DDP solver.
    *   Defines a specific cost function (`arm_cost_function`) for the arm, penalizing:
        *   Final end-effector position error.
        *   Deviation from a desired final state (e.g., zero velocities).
        *   Control effort (joint accelerations) throughout the trajectory.
    *   Runs the DDP optimization to find an optimal trajectory.
    *   Visualizes the results, including:
        *   Animation of the DDP-optimized arm movement.
        *   Plot of the DDP cost function convergence over iterations.
        *   Plots of optimized control inputs and state trajectories.
        *   Plot of the end-effector path.
    *   Includes a **naive trajectory generation method** for comparison, highlighting the benefits of DDP. This naive trajectory uses linear interpolation of joint angles to a predefined target configuration and a simple P-controller for execution.

## Differential Dynamic Programming (DDP) / iLQR

DDP is a powerful trajectory optimization algorithm based on dynamic programming and Newton's method. iLQR is a common variant that uses a linear approximation of the dynamics and a quadratic approximation of the cost function in the backward pass.

### Key Algorithm Steps:

1.  **Initialization:**
    *   Start with an initial guess for the control sequence `U = {u_0, ..., u_{N-1}}`.
    *   Perform an initial rollout: simulate the system dynamics forward using `U` to get a nominal state trajectory `X = {x_0, ..., x_N}` and calculate the total cost.

2.  **Iterative Refinement (until convergence or max iterations):**
    *   **a. Derivative Calculation:**
        *   Compute derivatives of the dynamics (`fx = ∂f/∂x`, `fu = ∂f/∂u`) along the current trajectory (X, U). This implementation uses numerical finite differences for these.
        *   Compute derivatives of the cost function (`lx = ∂l/∂x`, `lu = ∂l/∂u`, `lxx = ∂²l/∂x²`, etc.) along (X, U). This implementation expects the provided cost function to calculate these analytically.
    *   **b. Backward Pass:**
        *   Starting from the final time step `N` and moving backward to `k=0`.
        *   Compute quadratic approximations of the Action-Value function `Q(δx, δu)` at each step.
        *   `Q_x, Q_u, Q_xx, Q_uu, Q_ux` are computed using the cost derivatives and the value function derivatives from the next step (`V_x`, `V_xx`).
        *   Regularization is applied to `Q_uu` to ensure it's positive definite for inversion.
        *   Solve for the optimal control update policy: `δu_k = k_k + K_k δx_k`, where `k_k` is the feedforward term and `K_k` is the feedback gain matrix.
        *   Update the value function derivatives `V_x` and `V_xx` for the current step.
    *   **c. Forward Pass (with Line Search):**
        *   Starting from the initial state `x_0`.
        *   Apply the updated control policy: `u_k_new = u_k_old + α * k_k + K_k (x_k_new - x_k_old)`.
        *   `α` (alpha) is a line search parameter (0 < α ≤ 1) adjusted to ensure the new trajectory has a lower cost than the previous one.
        *   Simulate the system with the new controls to get `X_new`, `U_new`, and `cost_new`.
    *   **d. Update and Convergence Check:**
        *   If the line search is successful (cost improved), accept the new trajectory: `X = X_new`, `U = U_new`.
        *   If the cost improvement is below a tolerance `tol` (and `α=1.0`), or max iterations are reached, terminate.

## Project Structure

```
.
├── README.md         # This file (or README_updated.md)
└── src/
    ├── ddp.py        # Core DDP/iLQR algorithm implementation
    ├── robot_env.py  # 2-Link Arm simulation environment and visualizer
    ├── main.py       # Main script to run DDP for the arm and visualize
    └── __init__.py   # Makes src a package (optional)
```

## Requirements

*   Python 3.x
*   NumPy
*   Matplotlib

You can install the required libraries using pip:
```bash
pip install numpy matplotlib
```

## How to Run

1.  Navigate to the root directory of the project.
2.  Run the main script:
    ```bash
    python src/main.py
    ```

This will:
*   Initialize the 2-link arm and the DDP solver.
*   Run the DDP optimization process (console output will show iteration progress and costs).
*   Generate a naive trajectory for comparison.
*   Display Matplotlib windows/plots showing:
    *   An animation of the DDP-optimized robot arm movement.
    *   The DDP cost function evolution over iterations.
    *   Optimized control inputs (joint accelerations) over time.
    *   Optimized state trajectory (joint angles and velocities) over time.
    *   A comparison of the end-effector paths for the DDP-optimized trajectory and the naive trajectory, along with their respective costs.

If running in a headless environment (no display available), the script is configured to save the plots as PNG files (e.g., `ddp_cost_history.png`, `ddp_ee_path.png`, etc.) in the root directory. The animation itself might not display but a final frame might be saved.

## Customization

You can modify parameters in `src/main.py` to experiment:

*   **Robot Parameters (`LINK_LENGTHS`, `INITIAL_ANGLES_DEG`, `INITIAL_OMEGAS_RAD_S`, `DT`)**: Change the arm's physical properties or initial state.
*   **DDP Horizon (`HORIZON`)**: Affects the planning duration and computational cost.
*   **Target Position (`TARGET_EE_POS`)**: Change the goal for the end-effector.
*   **Cost Function Weights (`R_CONTROL_COST`, `Q_EE_FINAL_COST`, `Q_STATE_FINAL_COST`)**: These are crucial. Adjusting them will change the optimized behavior (e.g., prioritizing accuracy vs. smoothness/effort). Uncomment and tune optional running costs for different behaviors.
*   **DDP Solver Parameters (`max_iters`, `tol` in `ddp_solver.run(...)`)**: Control the DDP convergence criteria.
*   **Naive Trajectory Target (`TARGET_ANGLES_NAIVE_DEG`)**: Modify the target joint configuration for the naive comparison.

## Further Exploration

*   **Analytical Dynamics Derivatives:** Implement analytical Jacobians (`fx`, `fu`) for the `TwoLinkArm` dynamics in `ddp.py` or `robot_env.py` to significantly speed up DDP computation.
*   **More Complex Robot Models:** Extend to robots with more degrees of freedom or different dynamics (e.g., including full rigid body dynamics).
*   **Obstacle Avoidance:** Add obstacle information to the environment and incorporate penalties for collisions into the cost function.
*   **Different Cost Functions:** Experiment with other cost terms, such as minimizing joint velocities, tracking a specific end-effector orientation, or path following.
*   **Full DDP:** Implement the full DDP algorithm which also considers second-order derivatives of the dynamics (though iLQR is often sufficient and simpler).
*   **Control Limits:** Add explicit control input (acceleration/torque) limits within the DDP forward pass or by adding them to the cost function.
```
