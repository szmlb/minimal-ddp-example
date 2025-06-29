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

### Mathematical Formulation of iLQR

The goal of iLQR is to find a sequence of control inputs `U = {u_0, u_1, ..., u_{N-1}}` that minimizes a total cost `J` for a system with discrete-time dynamics.

**1. System Dynamics:**
The system evolves according to nonlinear dynamics:
`x_{k+1} = f(x_k, u_k)`
where `x_k` is the state at time step `k` and `u_k` is the control input at time step `k`.

**2. Cost Function:**
The total cost `J(X, U)` is the sum of running costs `l(x_k, u_k)` at each time step and a final terminal cost `l_f(x_N)`:
`J(X, U) = Σ_{k=0}^{N-1} l(x_k, u_k) + l_f(x_N)`
where `X = {x_0, ..., x_N}` is the state trajectory.

**3. Value Function:**
The optimal cost-to-go, or Value Function `V_k(x)`, is the minimum cost achievable starting from state `x` at time `k` until the final time `N`.
`V_k(x) = min_{u_k,...,u_{N-1}} [ Σ_{j=k}^{N-1} l(x_j, u_j) + l_f(x_N) ]`
The value function at the final step is simply the terminal cost:
`V_N(x_N) = l_f(x_N)`

**4. Action-Value Function (Q-function):**
The Q-function `Q_k(x, u)` is the cost of taking control `u` in state `x` at time `k`, and then following the optimal policy thereafter:
`Q_k(x, u) = l(x, u) + V_{k+1}(f(x, u))`

**5. Backward Pass:**
The backward pass starts from `k=N-1` down to `0`. At each step `k`, around the current nominal trajectory `(x̄_k, ū_k)`:
*   Approximate `V_{k+1}` quadratically around `x̄_{k+1}`:
    `V_{k+1}(x̄_{k+1} + δx) ≈ V_{k+1}(x̄_{k+1}) + V_x^T δx + 0.5 δx^T V_{xx} δx`
    where `V_x = ∂V_{k+1}/∂x` and `V_{xx} = ∂²V_{k+1}/∂x²` are evaluated at `x̄_{k+1}`.
*   The change in the Q-function, `δQ_k(δx_k, δu_k) = Q_k(x̄_k+δx_k, ū_k+δu_k) - Q_k(x̄_k, ū_k)`, is approximated by expanding `l(x,u)` to second order and `f(x,u)` to first order (iLQR assumption):
    `δx_{k+1} = f(x̄_k+δx_k, ū_k+δu_k) - f(x̄_k, ū_k) ≈ f_x δx_k + f_u δu_k`
    (where `f_x = ∂f/∂x` and `f_u = ∂f/∂u` are Jacobians evaluated at `(x̄_k, ū_k)`).
*   This leads to a quadratic approximation of `δQ_k`:
    `δQ_k ≈ 0.5 [δx_k^T, δu_k^T] Q̃ [δx_k; δu_k] + [Q_x^T, Q_u^T] [δx_k; δu_k]`
    (This is a general form; often expanded directly)
    More explicitly:
    `Q_x = l_x + f_x^T V_x'`  (prime `V_x'` denotes `V_x` from step `k+1`)
    `Q_u = l_u + f_u^T V_x'`
    `Q_{xx} = l_{xx} + f_x^T V_{xx}' f_x`  (iLQR ignores `V_x' f_{xx}` term from full DDP)
    `Q_{uu} = l_{uu} + f_u^T V_{xx}' f_u`  (iLQR ignores `V_x' f_{uu}` term)
    `Q_{ux} = l_{ux} + f_u^T V_{xx}' f_x`  (iLQR ignores `V_x' f_{ux}` term)
    (where `l_x, l_u, l_{xx}, l_{uu}, l_{ux}` are derivatives of the running cost `l(x_k, u_k)`)
*   The optimal control update `δu_k*` is found by minimizing this quadratic w.r.t `δu_k`: `∂(δQ_k)/∂(δu_k) = 0`.
    This gives: `δu_k* = -Q_{uu}^{-1} (Q_u + Q_{ux} δx_k)`
    This can be written as a linear feedback policy: `δu_k* = k_k + K_k δx_k`
    where:
    `k_k = -Q_{uu}^{-1} Q_u` (feedforward term)
    `K_k = -Q_{uu}^{-1} Q_{ux}` (feedback gain matrix)
    (Regularization `Q̃_{uu} = Q_{uu} + λI` is often used for `Q_{uu}` before inversion.)
*   The new value function derivatives for step `k` are updated:
    `V_x(k) = Q_x + K_k^T Q_{uu} k_k + K_k^T Q_u + Q_{ux}^T k_k` (or simpler forms like `V_x = Q_x - K_k^T Q_{uu} k_k`)
    `V_{xx}(k) = Q_{xx} + K_k^T Q_{uu} K_k + K_k^T Q_{ux} + Q_{ux}^T K_k` (or `V_{xx} = Q_{xx} - K_k^T Q_{uu} K_k`)

**6. Forward Pass:**
A new trajectory is simulated using the computed gains:
`u_k^{new} = ū_k + α k_k + K_k (x_k^{new} - x̄_k)`
`x_{k+1}^{new} = f(x_k^{new}, u_k^{new})`
A line search parameter `α` (0 < `α` ≤ 1) is used to scale the feedforward term `k_k` to ensure that the new trajectory results in a lower cost.

The process (Backward Pass -> Forward Pass) is iterated until the improvement in cost is below a threshold or a maximum number of iterations is reached.

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
