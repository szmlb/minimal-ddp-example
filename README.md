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

## Differential Dynamic Programming (DDP) and Iterative Linear Quadratic Regulator (iLQR)

Differential Dynamic Programming (DDP) is a trajectory optimization algorithm that uses dynamic programming and second-order approximations (like Newton's method) to find optimal controls for nonlinear systems. Iterative Linear Quadratic Regulator (iLQR) is a widely used variant of DDP.

**The core difference lies in how they approximate the dynamics:**
*   **Full DDP:** Uses a second-order Taylor expansion of the system dynamics $f(x,u)$ during the backward pass. This involves Hessians of the dynamics ($f_{xx}, f_{uu}, f_{ux}$).
*   **iLQR:** Simplifies this by using only a first-order Taylor expansion of the dynamics (i.e., linearization), thus ignoring the $f_{xx}, f_{uu}, f_{ux}$ terms. This makes computations simpler, as Hessians of dynamics can be complex to derive and compute.

**This project implements the iLQR variant.** The mathematical formulation below details iLQR.

### Mathematical Formulation (iLQR variant)

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

For **full DDP**, these $Q$-factor equations would additionally include terms involving second-order derivatives of the dynamics $f_{xx}, f_{uu}, f_{ux}$ (often contracted with $V_x'$). For example:
$Q_{xx}^{DDP} = l_{xx} + f_x^T V_{xx}' f_x + V_x' \cdot f_{xx}$
$Q_{uu}^{DDP} = l_{uu} + f_u^T V_{xx}' f_u + V_x' \cdot f_{uu}$
$Q_{ux}^{DDP} = l_{ux} + f_u^T V_{xx}' f_x + V_x' \cdot f_{ux}$
(where $V_x' \cdot f_{**}$ denotes an appropriate tensor contraction). **iLQR omits these $V_x' \cdot f_{**}$ terms.**

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

*   **Analytical Dynamics Derivatives:** Implement analytical Jacobians ($f_x, f_u$) for the `TwoLinkArm` dynamics in `src/robot_env.py` (or by modifying `src/ddp.py` to accept them). This would significantly speed up the DDP computation compared to the current numerical differentiation.
*   **Implement Full DDP:**
    *   Extend the current iLQR solver in `src/ddp.py` to optionally include second-order dynamics derivatives ($f_{xx}, f_{uu}, f_{ux}$) in the calculation of $Q_{xx}, Q_{uu}, Q_{ux}$ during the backward pass.
    *   This would require:
        *   A method to compute or be provided with these Hessians of dynamics (e.g., via numerical differentiation, though more complex, or analytically if the dynamics are simple enough).
        *   Modifying the `backward_pass` in `ddp.py` to incorporate these terms, likely controlled by a flag in the `DDP` class constructor (e.g., `is_full_ddp=True`).
        *   The $Q$-factor updates would become (conceptually, showing addition of new terms):
            $Q_{xx} \leftarrow Q_{xx} + V_x' \cdot f_{xx}$
            $Q_{uu} \leftarrow Q_{uu} + V_x' \cdot f_{uu}$
            $Q_{ux} \leftarrow Q_{ux} + V_x' \cdot f_{ux}$
            (where $V_x' \cdot f_{**}$ indicates an appropriate tensor contraction of the next state's value function gradient with the Hessians of the dynamics).
*   **More Complex Robot Models:** Adapt the framework for robots with more degrees of freedom or different types of dynamics (e.g., including full rigid body dynamics using libraries like Pinocchio or PyBullet).
*   **Obstacle Avoidance:** Integrate obstacle information into the `robot_env.py` and add penalty terms to the `arm_cost_function` to encourage collision-free paths.
*   **Different Cost Functions:** Experiment with a wider variety of cost terms, such as minimizing joint velocities/jerks, tracking a specific end-effector orientation, or following a predefined geometric path.
*   **Control Limits:** Implement stricter handling of control input limits (e.g., joint torque or acceleration limits). This can be done by adding them as constraints (more complex, often requiring augmented Lagrangian methods) or by penalizing violations in the cost function, or by clamping controls in the forward pass (as is done partially in the current DDP's forward pass comments).
```
