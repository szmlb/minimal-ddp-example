# iLQR for 2-Link Arm Motion Planning

This project demonstrates the application of the Iterative Linear Quadratic Regulator (iLQR) algorithm for optimizing the motion of a simple 2-link planar robotic arm. The goal is to drive the arm's end-effector to a target position while minimizing control effort and satisfying other objectives defined in a cost function. iLQR is a variant of Differential Dynamic Programming (DDP).

## Project Overview

The project includes:
1.  **A 2-Link Robotic Arm Simulation (`src/robot_env.py`):**
    *   Defines the arm's kinematics (`forward_kinematics`) and dynamics (`dynamics`).
    *   The state is `[theta1, theta2, omega1, omega2]` (joint angles and velocities).
    *   The control input is `[alpha1, alpha2]` (joint angular accelerations).
    *   Includes a `RobotVisualizer` class for animating the arm's movement using Matplotlib.
2.  **iLQR Implementation (`src/ddp.py`):**
    *   Contains the `iLQRSolver` class that iteratively optimizes a trajectory.
    *   Uses numerical differentiation for system dynamics Jacobians ($f_x, f_u$).
    *   Expects the cost function to provide analytical first and second derivatives ($l_x, l_u, l_{xx}, l_{uu}, l_{ux}$).
    *   Features a backward pass to compute control updates (feedforward and feedback gains) and a forward pass with line search to ensure cost reduction.
3.  **Main Application Script (`src/main.py`):**
    *   Sets up the 2-link arm environment and the `iLQRSolver`.
    *   Defines a specific cost function (`arm_cost_function`) for the arm, penalizing:
        *   Final end-effector position error.
        *   Deviation from a desired final state (e.g., zero velocities).
        *   Control effort (joint accelerations) throughout the trajectory.
    *   Runs the iLQR optimization to find an optimal trajectory.
    *   Visualizes the results, including:
        *   Animation of the iLQR-optimized arm movement.
        *   Plot of the iLQR cost function convergence over iterations.
        *   Plots of optimized control inputs and state trajectories.
        *   Plot of the end-effector path.
    *   Includes a **naive trajectory generation method** for comparison, highlighting the benefits of iLQR. This naive trajectory uses linear interpolation of joint angles to a predefined target configuration and a simple P-controller for execution.

## Iterative Linear Quadratic Regulator (iLQR) Algorithm

The Iterative Linear Quadratic Regulator (iLQR) is a powerful trajectory optimization algorithm well-suited for systems with nonlinear dynamics. It iteratively linearizes the system dynamics and quadratically approximates the cost function around a nominal trajectory to find locally optimal control sequences.

**Relationship to Differential Dynamic Programming (DDP):** iLQR is a variant of DDP. The primary simplification in iLQR compared to full DDP is that iLQR only uses a first-order (linear) approximation of the system dynamics, thereby avoiding the computation of second-order derivatives of the dynamics (Hessians like $f_{xx}, f_{uu}, f_{ux}$). This often makes iLQR easier to implement and computationally less expensive, while still providing excellent results for many problems.

**The `iLQRSolver` class in `src/ddp.py` implements this iLQR algorithm.**

### Core Mathematical Formulation of iLQR

The goal of iLQR is to find a sequence of control inputs $U = \{u_0, u_1, ..., u_{N-1}\}$ that minimizes a total cost $J$ for a system with discrete-time dynamics.

**1. System Dynamics:**
The system evolves according to nonlinear dynamics:
$$x_{k+1} = f(x_k, u_k)$$
where $x_k$ is the state at time step $k$ and $u_k$ is the control input at time step $k$.
For iLQR, we linearize these dynamics around the nominal trajectory $(\bar{x}_k, \bar{u}_k)$:
$$\delta x_{k+1} \approx f_x \delta x_k + f_u \delta u_k$$
where $\delta x_k = x_k - \bar{x}_k$, $\delta u_k = u_k - \bar{u}_k$, $f_x = \frac{\partial f}{\partial x}(\bar{x}_k, \bar{u}_k)$, and $f_u = \frac{\partial f}{\partial u}(\bar{x}_k, \bar{u}_k)$.

**2. Cost Function:**
The total cost $J(X, U)$ is the sum of running costs $l(x_k, u_k)$ at each time step and a final terminal cost $l_f(x_N)$:
$$J(X, U) = \sum_{k=0}^{N-1} l(x_k, u_k) + l_f(x_N)$$
where $X = \{x_0, ..., x_N\}$ is the state trajectory. We use a quadratic approximation of the cost around the nominal trajectory.

**3. Value Function:**
The optimal cost-to-go, or Value Function $V_k(x)$, is the minimum cost achievable starting from state $x$ at time $k$ until the final time $N$.
$$V_k(x) = \min_{u_k,...,u_{N-1}} \left[ \sum_{j=k}^{N-1} l(x_j, u_j) + l_f(x_N) \right]$$
The value function at the final step is simply the terminal cost:
$$V_N(x_N) = l_f(x_N)$$
Its quadratic approximation around $\bar{x}_N$ is:
$$V_N(\bar{x}_N + \delta x_N) \approx V_N(\bar{x}_N) + l_{f,x}^T \delta x_N + \frac{1}{2} \delta x_N^T l_{f,xx} \delta x_N$$
So, $V_{N,x} = l_{f,x}$ and $V_{N,xx} = l_{f,xx}$.

**4. Action-Value Function (Q-function):**
The Q-function $Q_k(x, u)$ is the cost of taking control $u$ in state $x$ at time $k$, and then following the optimal policy thereafter:
$$Q_k(x, u) = l(x, u) + V_{k+1}(f(x, u))$$

**5. Backward Pass:**
The backward pass iterates from $k=N-1$ down to $0$. At each step $k$, we approximate $V_{k+1}$ quadratically around $\bar{x}_{k+1}$:
$$V_{k+1}(\bar{x}_{k+1} + \delta x) \approx V_{k+1}(\bar{x}_{k+1}) + V_{k+1,x}^T \delta x + \frac{1}{2} \delta x^T V_{k+1,xx} \delta x$$
The local quadratic model of the Q-function around $(\bar{x}_k, \bar{u}_k)$ for deviations $(\delta x_k, \delta u_k)$ is:
$$\delta Q_k \approx \frac{1}{2} \begin{bmatrix} \delta x_k \\ \delta u_k \end{bmatrix}^T \begin{bmatrix} Q_{xx} & Q_{ux}^T \\ Q_{ux} & Q_{uu} \end{bmatrix} \begin{bmatrix} \delta x_k \\ \delta u_k \end{bmatrix} + \begin{bmatrix} Q_x^T & Q_u^T \end{bmatrix} \begin{bmatrix} \delta x_k \\ \delta u_k \end{bmatrix}$$
The derivatives of $Q_k$ (the $Q$-factors) are (with $V_x'$ and $V_{xx}'$ denoting derivatives of $V_{k+1}$):
$$Q_x = l_x + f_x^T V_x'$$
$$Q_u = l_u + f_u^T V_x'$$
$$Q_{xx} = l_{xx} + f_x^T V_{xx}' f_x$$
$$Q_{uu} = l_{uu} + f_u^T V_{xx}' f_u$$
$$Q_{ux} = l_{ux} + f_u^T V_{xx}' f_x$$
(Note: $l_x, l_u, l_{xx}, l_{uu}, l_{ux}$ are derivatives of the running cost $l(x_k, u_k)$.)

The optimal control update $\delta u_k^*$ is found by minimizing the quadratic model of $Q_k$ w.r.t $\delta u_k$, which means setting $\frac{\partial (\delta Q_k)}{\partial (\delta u_k)} = Q_u + Q_{uu}\delta u_k + Q_{ux}\delta x_k = 0$.
This gives the control policy update: $\delta u_k^* = k_k + K_k \delta x_k$, where:
$$k_k = -Q_{uu}^{-1} Q_u \quad \text{(feedforward term)}$$
$$K_k = -Q_{uu}^{-1} Q_{ux} \quad \text{(feedback gain matrix)}$$
(Regularization, e.g., $Q_{uu,reg} = Q_{uu} + \lambda I$, is often applied to $Q_{uu}$ before inversion.)

The new value function derivatives for step $k$ are updated:
$$V_{k,x} = Q_x - K_k^T Q_{uu} k_k$$
$$V_{k,xx} = Q_{xx} - K_k^T Q_{uu} K_k$$

**6. Forward Pass:**
A new trajectory is simulated using the computed gains:
$$u_k^{\text{new}} = \bar{u}_k + \alpha k_k + K_k (x_k^{\text{new}} - \bar{x}_k)$$
$$x_{k+1}^{\text{new}} = f(x_k^{\text{new}}, u_k^{\text{new}})$$
A line search parameter $\alpha \in (0, 1]$ is used to scale the feedforward term $k_k$ to ensure the new trajectory results in a lower cost.

The process (Backward Pass $\rightarrow$ Forward Pass) is iterated until cost improvement is below a threshold or max iterations are reached.

### Key Algorithm Steps:

1.  **Initialization:**
    *   Start with an initial guess for the control sequence `U = {u_0, ..., u_{N-1}}`.
    *   Perform an initial rollout: simulate the system dynamics forward using `U` to get a nominal state trajectory `X = {x_0, ..., x_N}` and calculate the total cost.

2.  **Iterative Refinement (until convergence or max iterations):**
    *   **a. Derivative Calculation:**
    *   **a. Derivative Calculation:**
        *   Compute Jacobians of the dynamics ($f_x = \frac{\partial f}{\partial x}$, $f_u = \frac{\partial f}{\partial u}$) along the current trajectory $(X, U)$. This implementation uses numerical finite differences.
        *   Compute derivatives of the cost function ($l_x, l_u, l_{xx}, l_{uu}, l_{ux}$) along $(X, U)$. This implementation expects the provided cost function to calculate these analytically.
    *   **b. Backward Pass:** (Described in detail in the "Core Mathematical Formulation" section above)
        *   Starting from the final time step $N$ and moving backward to $k=0$.
        *   Compute $Q$-function derivatives ($Q_x, Q_u, Q_{xx}, Q_{uu}, Q_{ux}$).
        *   Calculate feedforward ($k_k$) and feedback ($K_k$) terms for the control update.
        *   Update Value function derivatives ($V_{k,x}, V_{k,xx}$).
    *   **c. Forward Pass (with Line Search):**
        *   Starting from the initial state $x_0$.
        *   Apply the updated control policy: $u_k^{\text{new}} = \bar{u}_k + \alpha k_k + K_k (x_k^{\text{new}} - \bar{x}_k)$.
        *   $\alpha$ (alpha) is a line search parameter ($0 < \alpha \le 1$) adjusted to ensure the new trajectory has a lower cost.
        *   Simulate the system with the new controls to get $X_{\text{new}}, U_{\text{new}}$, and $cost_{\text{new}}$.
    *   **d. Update and Convergence Check:**
        *   If line search is successful (cost improved), accept the new trajectory: $X = X_{\text{new}}, U = U_{\text{new}}$.
        *   If cost improvement is below a tolerance `tol` (and $\alpha=1.0$), or max iterations are reached, terminate.

## Project Structure

```
.
├── README.md         # This file
└── src/
    ├── ddp.py        # Contains iLQRSolver and a conceptual sketch for FullDDPSolver
    ├── robot_env.py  # 2-Link Arm simulation environment and visualizer
    ├── main.py       # Main script to run iLQR for the arm and visualize
    └── __init__.py   # Makes src a package (optional)
```

## Conceptual FullDDPSolver Sketch

The file `src/ddp.py` also contains a class `FullDDPSolver` which is a conceptual sketch inheriting from `iLQRSolver`. It outlines where modifications would be needed to implement full DDP, primarily:
*   **Dynamics Hessians:** It includes a placeholder method `_get_dynamics_hessians` that would need to compute or be provided with $f_{xx}, f_{uu}, f_{ux}$.
*   **Backward Pass:** The `backward_pass` method is overridden to show where terms involving these Hessians (contracted with $V_x'$) would be added to the $Q$-factor calculations ($Q_{xx}, Q_{uu}, Q_{ux}$).

This sketch is for illustrative purposes to show the structural difference from iLQR and is not a fully operational DDP solver without a concrete implementation for computing and utilizing the dynamics Hessians.

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
*   Initialize the 2-link arm and the `iLQRSolver`.
*   Run the iLQR optimization process (console output will show iteration progress and costs).
*   Generate a naive trajectory for comparison.
*   Display Matplotlib windows/plots showing:
    *   An animation of the iLQR-optimized robot arm movement.
    *   The iLQR cost function evolution over iterations.
    *   Optimized control inputs (joint accelerations) over time.
    *   Optimized state trajectory (joint angles and velocities) over time.
    *   A comparison of the end-effector paths for the iLQR-optimized trajectory and the naive trajectory, along with their respective costs.

If running in a headless environment (no display available), the script is configured to save the plots as PNG files (e.g., `ilqr_cost_history.png`, `ilqr_ee_path.png`, etc.) in the root directory. The animation itself might not display but a final frame might be saved.

## Customization

You can modify parameters in `src/main.py` to experiment:

*   **Robot Parameters (`LINK_LENGTHS`, `INITIAL_ANGLES_DEG`, `INITIAL_OMEGAS_RAD_S`, `DT`)**: Change the arm's physical properties or initial state.
*   **iLQR Horizon (`HORIZON`)**: Affects the planning duration and computational cost.
*   **Target Position (`TARGET_EE_POS`)**: Change the goal for the end-effector.
*   **Cost Function Weights (`R_CONTROL_COST`, `Q_EE_FINAL_COST`, `Q_STATE_FINAL_COST`)**: These are crucial. Adjusting them will change the optimized behavior (e.g., prioritizing accuracy vs. smoothness/effort). Uncomment and tune optional running costs for different behaviors.
*   **iLQR Solver Parameters (`max_iters`, `tol` in `ilqr_solver.run(...)`)**: Control the iLQR convergence criteria.
*   **Naive Trajectory Target (`TARGET_ANGLES_NAIVE_DEG`)**: Modify the target joint configuration for the naive comparison.

## Further Exploration & CasADi Integration

A significant enhancement to this project would be the integration of a symbolic math framework like **CasADi** for automatic differentiation (AD).

**Benefits of CasADi:**
*   **Exact Derivatives:** Automatically compute exact Jacobians and Hessians for dynamics and cost functions, eliminating manual derivation and numerical inaccuracies.
*   **Speed:** CasADi can generate efficient C code for derivative calculations, leading to faster solver iterations.
*   **Flexibility:** Simplifies modifications to robot models or cost functions, as derivatives are re-derived automatically.

**Conceptual CasADi Integration Steps:**
1.  **Symbolic Definitions:** Rewrite the `TwoLinkArm` dynamics and `arm_cost_function` using CasADi's symbolic variables and operations.
2.  **AD for Derivatives:**
    *   Use `casadi.jacobian` for $f_x, f_u$.
    *   Use `casadi.gradient` and `casadi.hessian` for $l_x, l_u, l_{xx}, l_{uu}, l_{ux}$ (and terminal cost derivatives).
    *   For full DDP, CasADi would also compute Hessians of dynamics ($f_{xx}, f_{uu}, f_{ux}$), typically by taking Hessians of each component of the dynamics vector $f$.
3.  **Callable Functions:** Convert these symbolic derivative expressions into callable Python functions using `casadi.Function`.
4.  **Solver Modification:**
    *   The `iLQRSolver` (and `FullDDPSolver`) would be modified (as partially done in `src/ddp.py`) to accept these callable derivative functions during initialization.
    *   The internal `_compute_derivatives` method would then use these pre-compiled functions instead of numerical differentiation or expecting the cost function to return derivatives directly.
    *   The `src/main.py` script includes a conceptual (commented-out) block demonstrating how these symbolic definitions and CasADi function generation would look for the `TwoLinkArm`. The `if __name__ == '__main__'` block in `src/ddp.py` also contains a working example of using CasADi for the simpler 1D system.

**Other Potential Enhancements:**
*   **Analytical Dynamics Derivatives (without CasADi):** Manually derive and implement analytical Jacobians ($f_x, f_u$) for the `TwoLinkArm` dynamics.
*   **Complete Full DDP Implementation:** Fully implement the computation (e.g., via CasADi or numerical Hessians) and utilization of second-order dynamics derivatives in the `FullDDPSolver`.
*   **More Complex Robot Models:** Adapt for robots with more DoFs or using rigid body dynamics libraries.
*   **Obstacle Avoidance:** Add collision penalties to the cost function.
*   **Varied Cost Functions:** Explore costs for minimizing jerk, tracking orientation, etc.
*   **Control Limits:** Implement robust handling of control input saturation.
```
