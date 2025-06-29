import numpy as np

class iLQRSolver:
    def __init__(self, dynamics_fn, cost_fn, state_dim, control_dim, horizon,
                 fx_func=None, fu_func=None, cost_derivatives_func=None):
        """
        Iterative Linear Quadratic Regulator (iLQR) Solver.
        Can use numerical differentiation for dynamics if specific derivative functions are not provided.

        Args:
            dynamics_fn (callable): Function `next_x = f(x, u)` for system simulation.
            cost_fn (callable): Function `cost = l(x, u, k, is_final)` for scalar cost.
                                OR, if cost_derivatives_func is None, this function should be
                                `cost, lx, lu, lxx, luu, lux = l(x, u, k, is_final, get_derivatives=True)`.
            state_dim (int): Dimension of the state vector `x`.
            control_dim (int): Dimension of the control input vector.
            horizon (int): Number of time steps in the trajectory (N).
            fx_func (callable, optional): CasADi-generated function for Jacobian of dynamics wrt x.
                                         Signature: fx_matrix = fx_func(x, u).
            fu_func (callable, optional): CasADi-generated function for Jacobian of dynamics wrt u.
                                         Signature: fu_matrix = fu_func(x, u).
            cost_derivatives_func (callable, optional):
                CasADi-generated function for cost and its derivatives.
                Signature: cost, lx, lu, lxx, luu, lux = func(x, u, is_final_flag)
                           where is_final_flag is 0 for running, 1 for terminal.
                           (Terminal cost ignores u, so pass dummy u like np.zeros(control_dim)).
        """
        self.dynamics_fn = dynamics_fn # Used for rollouts
        self.cost_fn = cost_fn         # Used for scalar cost in rollouts if cost_derivatives_func is provided,
                                       # otherwise used for derivatives too.
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.N = horizon

        self.fx_func = fx_func
        self.fu_func = fu_func
        self.cost_derivatives_func = cost_derivatives_func

        if self.fx_func is None or self.fu_func is None:
            print("Warning: Dynamics Jacobians (fx_func, fu_func) not provided. "
                  "iLQRSolver will use numerical differentiation for dynamics.")
        if self.cost_derivatives_func is None:
            print("Warning: cost_derivatives_func not provided. "
                  "Falling back to cost_fn to provide derivatives.")

        # Regularization parameters
        self.reg_factor = 10
        self.reg_min = 1e-6
        self.reg_max = 1e10

    def _compute_derivatives(self, X, U):
        """
        Compute derivatives for the iLQR backward pass.
        Uses CasADi-generated functions if provided, otherwise falls back to
        numerical differentiation for dynamics and expects cost_fn to provide its derivatives.
        """
        fx_list = []
        fu_list = []
        lx_list = []
        lu_list = []
        lxx_list = []
        luu_list = []
        lux_list = []

        for k in range(self.N):
            x_k = X[k]
            u_k = U[k]

            # Dynamics derivatives
            if self.fx_func and self.fu_func:
                fx_val = self.fx_func(x_k, u_k)
                fu_val = self.fu_func(x_k, u_k)
                # Ensure they are numpy arrays if coming from CasADi
                fx_list.append(np.asarray(fx_val))
                fu_list.append(np.asarray(fu_val))
            else: # Fallback to numerical differentiation
                Fx_k_num, Fu_k_num = self._get_dynamics_jacobians(x_k, u_k)
                fx_list.append(Fx_k_num)
                fu_list.append(Fu_k_num)

            # Cost derivatives
            if self.cost_derivatives_func:
                # is_final_flag = 0 for running cost
                _, lx_k_val, lu_k_val, lxx_k_val, luu_k_val, lux_k_val = \
                    self.cost_derivatives_func(x_k, u_k, 0)
                lx_list.append(np.asarray(lx_k_val).flatten())
                lu_list.append(np.asarray(lu_k_val).flatten())
                lxx_list.append(np.asarray(lxx_k_val))
                luu_list.append(np.asarray(luu_k_val))
                lux_list.append(np.asarray(lux_k_val))
            else: # Fallback to original cost_fn
                _, lx_k_orig, lu_k_orig, lxx_k_orig, luu_k_orig, lux_k_orig = \
                    self.cost_fn(x_k, u_k, k, is_final=False, get_derivatives=True)
                lx_list.append(lx_k_orig)
                lu_list.append(lu_k_orig)
                lxx_list.append(lxx_k_orig)
                luu_list.append(luu_k_orig)
                lux_list.append(lux_k_orig)

        # Terminal cost derivatives
        x_N = X[self.N]
        if self.cost_derivatives_func:
            # is_final_flag = 1 for terminal cost, pass dummy control
            _, lx_N_val, _, lxx_N_val, _, _ = \
                self.cost_derivatives_func(x_N, np.zeros(self.control_dim), 1)
            lx_N_final = np.asarray(lx_N_val).flatten()
            lxx_N_final = np.asarray(lxx_N_val)
        else:
            _, lx_N_orig, _, lxx_N_orig, _, _ = \
                self.cost_fn(x_N, np.zeros(self.control_dim), self.N, is_final=True, get_derivatives=True)
            lx_N_final = lx_N_orig
            lxx_N_final = lxx_N_orig

        derivatives = {
            'fx': fx_list, 'fu': fu_list,
            'lx': lx_list, 'lu': lu_list,
            'lxx': lxx_list, 'luu': luu_list, 'lux': lux_list,
            'lx_N': lx_N, 'lxx_N': lxx_N
        }
        return derivatives

    def _get_dynamics_jacobians(self, x, u, delta=1e-5):
        """
        Numerically compute Jacobians of the dynamics function f(x, u)
        using central finite differences.
        Fx = ∂f/∂x
        Fu = ∂f/∂u

        Args:
            x (np.array): Current state vector.
            u (np.array): Current control vector.
            delta (float): Perturbation size for finite differences.

        Returns:
            tuple: (Fx, Fu)
                Fx (np.array): Jacobian of dynamics wrt state (state_dim, state_dim).
                Fu (np.array): Jacobian of dynamics wrt control (state_dim, control_dim).
        """
        # Compute Fx = ∂f/∂x
        Fx = np.zeros((self.state_dim, self.state_dim))
        x_perturbed = np.copy(x) # Ensure x is float for perturbation
        for i in range(self.state_dim):
            original_xi = x_perturbed[i]
            x_perturbed[i] = original_xi + delta
            f_plus = self.dynamics_fn(x_perturbed, u)
            x_perturbed[i] = original_xi - delta
            f_minus = self.dynamics_fn(x_perturbed, u)
            Fx[:, i] = (f_plus - f_minus) / (2 * delta)
            x_perturbed[i] = original_xi # Reset perturbation

        # Compute Fu = ∂f/∂u
        Fu = np.zeros((self.state_dim, self.control_dim))
        u_perturbed = np.copy(u) # Ensure u is float for perturbation
        for i in range(self.control_dim):
            original_ui = u_perturbed[i]
            u_perturbed[i] = original_ui + delta
            f_plus = self.dynamics_fn(x, u_perturbed)
            u_perturbed[i] = original_ui - delta
            f_minus = self.dynamics_fn(x, u_perturbed)
            Fu[:, i] = (f_plus - f_minus) / (2 * delta)
            u_perturbed[i] = original_ui # Reset perturbation

        return Fx, Fu

    def backward_pass(self, X, U, derivatives):
        """
        Performs the backward pass of the iLQR algorithm.
        Computes the optimal control policy update (feedforward k_ff and feedback K_fb terms)
        by quadratically approximating the action-value function Q and minimizing it.

        Args:
            X (np.array): Current state trajectory.
            U (np.array): Current control trajectory.
            derivatives (dict): Dictionary containing dynamics and cost derivatives.

        Returns:
            tuple: (k_feedforward, K_feedback, success_flag)
                k_feedforward (list): List of feedforward control terms.
                K_feedback (list): List of feedback gain matrices.
                success_flag (bool): True if backward pass was successful, False otherwise.
        """
        V_x = derivatives['lx_N']  # Value function gradient at terminal state
        V_xx = derivatives['lxx_N'] # Value function Hessian at terminal state

        k_feedforward_terms = [] # List to store feedforward terms k_k
        K_feedback_terms = []    # List to store feedback gain matrices K_k

        # Initial regularization parameter for this backward pass
        # This can be adjusted based on success/failure of matrix inversion
        current_regularization = self.reg_factor

        for k in range(self.N - 1, -1, -1): # Iterate backwards from N-1 down to 0
            lx = derivatives['lx'][k]
            lu = derivatives['lu'][k]
            lxx = derivatives['lxx'][k]
            luu = derivatives['luu'][k]
            lux = derivatives['lux'][k]
            fx = derivatives['fx'][k]
            fu = derivatives['fu'][k]

            # Q-function expansion terms (derivatives of Q function)
            # Q(x,u) = l(x,u) + V'(f(x,u))
            # V' is value function at next step (k+1), so V_x and V_xx are V_{k+1,x} and V_{k+1,xx}

            Q_x = lx + fx.T @ V_x
            Q_u = lu + fu.T @ V_x

            # For iLQR, we use only first-order derivatives of dynamics (fx, fu)
            Q_xx = lxx + fx.T @ V_xx @ fx
            Q_uu = luu + fu.T @ V_xx @ fu
            Q_ux = lux + fu.T @ V_xx @ fx

            # For full DDP, second-order derivatives of dynamics (f_xx, f_uu, f_ux) are also needed:
            # These terms would be added if self.is_full_ddp (assuming such a flag exists):
            #   f_xx_k, f_uu_k, f_ux_k = self._get_dynamics_hessians(X[k], U[k]) # Hypothetical function
            #   Q_xx += np.einsum('i,ijk->jk', V_x, f_xx_k) # Example of tensor contraction
            #   Q_uu += np.einsum('i,ijk->jk', V_x, f_uu_k) # Example of tensor contraction
            #   Q_ux += np.einsum('i,ijk->jk', V_x, f_ux_k) # Example of tensor contraction
            # Note: The exact form of contraction depends on how f_xx, f_uu, f_ux (tensors) are structured.

            # Regularization for Quu to ensure positive definiteness for inversion.
            # This is a common strategy to handle non-convexity or poor conditioning.
            Q_uu_reg = Q_uu + np.eye(self.control_dim) * current_regularization

            # Attempt to compute Cholesky decomposition of Quu_reg.
            # If it fails, Quu_reg is not positive definite. Increase regularization and retry.
            # This loop helps find a suitable regularization value.
            max_reg_increases = 10 # Prevent infinite loop if regularization keeps failing
            for _ in range(max_reg_increases):
                try:
                    # Cholesky decomposition is a good test for positive definiteness
                    # and can be used to solve the linear system Quu_reg * update = -Qu
                    np.linalg.cholesky(Q_uu_reg)
                    break # Success
                except np.linalg.LinAlgError:
                    # Increase regularization
                    current_regularization = max(self.reg_min, current_regularization * self.reg_factor)
                    current_regularization = min(current_regularization, self.reg_max)
                    Q_uu_reg = Q_uu + np.eye(self.control_dim) * current_regularization
                    print(f"    Warning: Quu not positive definite at k={k}. Increased regularization to {current_regularization:.2e}")
            else: # If loop completes without break (max_reg_increases reached)
                print(f"    ERROR: Quu not positive definite at k={k} after {max_reg_increases} regularization increases. Max reg: {current_regularization:.2e}. Aborting backward pass.")
                return None, None, False # Indicate failure of backward pass

            # Solve for feedforward (k_k) and feedback (K_k) terms
            # δu = -Quu_reg^{-1} * Qu  (feedforward term, k_k)
            #      -Quu_reg^{-1} * Qux * δx (feedback term, K_k * δx)
            # Using solve instead of inv for better numerical stability:
            try:
                k_k = np.linalg.solve(Q_uu_reg, -Q_u)
                K_k = np.linalg.solve(Q_uu_reg, -Q_ux)
            except np.linalg.LinAlgError: # Should be rare if Cholesky succeeded
                 print(f"    ERROR: Solving for k_k, K_k failed at k={k} even after regularization. Aborting.")
                 return None, None, False


            k_feedforward_terms.insert(0, k_k) # Prepend to maintain order
            K_feedback_terms.insert(0, K_k)   # Prepend

            # Update Value function derivatives for the previous time step (k-1)
            # V_x  = Q_x + K_k.T @ Q_uu @ k_k + K_k.T @ Q_u + Q_ux.T @ k_k
            # V_xx = Q_xx + K_k.T @ Q_uu @ K_k + K_k.T @ Q_ux + Q_ux.T @ K_k
            # These are standard iLQR update rules.
            V_x = Q_x + K_k.T @ Q_uu_reg @ k_k + K_k.T @ Q_u + Q_ux.T @ k_k # Using Quu_reg for consistency
            # V_x = Q_x - K_k.T @ Q_uu_reg @ k_k # A common simplified form
            V_xx = Q_xx + K_k.T @ Q_uu_reg @ K_k + K_k.T @ Q_ux + Q_ux.T @ K_k
            V_xx = 0.5 * (V_xx + V_xx.T) # Ensure V_xx remains symmetric

        return k_feedforward_terms, K_feedback_terms, True # Success

    def forward_pass(self, X_old, U_old, k_feedforward, K_feedback, alpha_line_search=1.0):
        """
        Performs the forward pass to compute the new trajectory (X_new, U_new)
        using the computed feedforward (k_feedforward) and feedback (K_feedback) terms.
        A line search (parameterized by `alpha_line_search`) is applied to the feedforward term
        to ensure cost reduction.

        Args:
            X_old (np.array): Previous state trajectory.
            U_old (np.array): Previous control trajectory.
            k_feedforward (list): List of feedforward control terms from backward pass.
            K_feedback (list): List of feedback gain matrices from backward pass.
            alpha_line_search (float): Line search parameter (0 < alpha <= 1).

        Returns:
            tuple: (X_new, U_new, total_cost_new)
                X_new (np.array): New state trajectory.
                U_new (np.array): New control trajectory.
                total_cost_new (float): Total cost of the new trajectory.
        """
        X_new = np.zeros_like(X_old)
        U_new = np.zeros_like(U_old)
        X_new[0] = X_old[0] # Initial state is fixed and does not change

        total_cost_new = 0.0

        for k in range(self.N): # Iterate from 0 to N-1
            # Control update rule: u_new[k] = u_old[k] + alpha * k_feedforward[k] + K_feedback[k] * (x_new[k] - x_old[k])
            delta_x_k = X_new[k] - X_old[k] # Difference from nominal trajectory state
            delta_u_k = alpha_line_search * k_feedforward[k] + K_feedback[k] @ delta_x_k

            U_new[k] = U_old[k] + delta_u_k

            # Optional: Apply control limits if defined for the system
            # U_new[k] = np.clip(U_new[k], self.control_min, self.control_max)

            # Simulate the system dynamics forward with the new control U_new[k]
            # to get the next state X_new[k+1]
            X_new[k+1] = self.dynamics_fn(X_new[k], U_new[k])

            # Accumulate running cost
            # Note: cost_fn is called with get_derivatives=False as we only need the scalar cost here
            cost_k, _, _, _, _, _ = self.cost_fn(X_new[k], U_new[k], k, is_final=False, get_derivatives=False)
            total_cost_new += cost_k

        # Add terminal cost for the final state X_new[N]
        # A dummy control (e.g., zeros) is passed as it's usually not part of terminal cost.
        term_cost_k, _, _, _, _, _ = self.cost_fn(X_new[self.N], np.zeros(self.control_dim), self.N, is_final=True, get_derivatives=False)
        total_cost_new += term_cost_k

        return X_new, U_new, total_cost_new

    def run(self, x0, U_initial_guess, max_iters=100, tol=1e-4):
        """
        """
        Run the main iLQR algorithm loop.

        Args:
            x0 (np.array): Initial state of the system (state_dim,).
            U_initial_guess (np.array): Initial guess for the control sequence (N, control_dim).
            max_iters (int): Maximum number of iLQR iterations.
            tol (float): Tolerance for cost improvement to determine convergence.
                         Convergence is declared if `cost_improvement < tol` and line search
                         succeeded with `alpha = 1.0`.

        Returns:
            tuple: (X_optimal, U_optimal, final_cost, cost_history_list)
                X_optimal (np.array): Optimal state trajectory (N+1, state_dim).
                U_optimal (np.array): Optimal control sequence (N, control_dim).
                final_cost (float): Cost of the optimal trajectory.
                cost_history_list (list): List of total costs at each iteration.
        """
        X = np.zeros((self.N + 1, self.state_dim)) # State trajectory
        U = np.copy(U_initial_guess)               # Control trajectory
        X[0] = x0                                  # Set initial state

        cost_history = [] # To store cost at each iteration

        # --- Initial Rollout and Cost Calculation ---
        # Simulate the system with the initial control guess to get the initial state trajectory
        # and calculate its total cost.
        current_X_trajectory = np.copy(X)
        current_total_cost = 0.0
        for k in range(self.N):
            current_X_trajectory[k+1] = self.dynamics_fn(current_X_trajectory[k], U[k])
            # Cost calculation (derivatives not needed for initial rollout cost)
            cost_k, _, _, _, _, _ = self.cost_fn(current_X_trajectory[k], U[k], k, is_final=False, get_derivatives=False)
            current_total_cost += cost_k
        # Add terminal cost
        term_cost_k, _, _, _, _, _ = self.cost_fn(current_X_trajectory[self.N], np.zeros(self.control_dim), self.N, is_final=True, get_derivatives=False)
        current_total_cost += term_cost_k

        cost_history.append(current_total_cost)
        print(f"Initial Cost: {current_total_cost:.4f}")

        # Set X to be the result of this initial rollout for the first iteration
        X = np.copy(current_X_trajectory)

        # --- Main iLQR Iteration Loop ---
        for i in range(max_iters):
            print(f"Iteration {i+1}/{max_iters}")

            # 1. Compute Derivatives (Backward Pass Preparation)
            #    Calculates Jacobians of dynamics (fx, fu) and
            #    derivatives of the cost function (lx, lu, lxx, luu, lux)
            #    along the current trajectory (X, U).
            # print("  Computing derivatives...") # Can be verbose
            derivatives = self._compute_derivatives(X, U)
            # `derivatives` is a dictionary containing all necessary derivative terms.

            # 2. Backward Pass
            #    Computes the optimal control update (feedforward k_ff and feedback K_fb terms)
            #    by working backwards from the final time step.
            # print("  Performing backward pass...") # Can be verbose
            k_ff_terms, K_fb_terms, backward_pass_successful = self.backward_pass(X, U, derivatives)

            if not backward_pass_successful:
                print("  Backward pass failed (e.g., Quu not positive definite even with regularization). Stopping.")
                # A more robust implementation might try increasing global regularization factor `self.reg_factor`
                # or other recovery strategies before stopping.
                # self.reg_factor = min(self.reg_factor * 10, self.reg_max) # Example: increase global reg
                return X, U, current_total_cost, cost_history # Return current best if BP fails


            # 3. Forward Pass with Line Search
            #    Applies the computed control updates and simulates the system forward.
            #    Uses a line search on the feedforward term (alpha) to ensure cost reduction.
            # print("  Performing forward pass with line search...") # Can be verbose
            alpha = 1.0  # Start with full step for line search
            min_alpha = 1e-8 # Minimum step size for line search
            line_search_reduction_factor = 0.5 # Factor to reduce alpha if cost does not improve

            while alpha >= min_alpha:
                X_new, U_new, new_total_cost = self.forward_pass(X, U, k_ff_terms, K_fb_terms, alpha_line_search=alpha)

                if new_total_cost < current_total_cost:
                    # print(f"    Line search success: alpha={alpha:.2e}, New Cost={new_total_cost:.4f} (Old: {current_total_cost:.4f})")
                    break # Found a trajectory with lower cost
                else:
                    # print(f"    Line search: alpha={alpha:.2e}, New Cost={new_total_cost:.4f} >= Old Cost={current_total_cost:.4f}. Reducing alpha.")
                    alpha *= line_search_reduction_factor
            else: # Executed if the while loop finishes without a 'break' (line search failed)
                print("  Line search failed to improve cost even with minimum alpha. Stopping.")
                # This might indicate issues with derivatives, cost function, or extreme non-convexity.
                return X, U, current_total_cost, cost_history # Return current best

            # --- Update and Convergence Check ---
            cost_improvement = current_total_cost - new_total_cost
            current_total_cost = new_total_cost
            X = X_new
            U = U_new
            cost_history.append(current_total_cost) # Record cost for this iteration

            print(f"  Iteration {i+1} complete. Cost: {current_total_cost:.4f}, Improvement: {cost_improvement:.4f}, Alpha: {alpha:.2e}")

            # Convergence check:
            # Improvement is small, AND the line search took a full step (alpha=1.0).
            # If alpha is small, it means the quadratic model might not be a good fit,
            # so convergence might be premature if only checking cost_improvement.
            if cost_improvement < tol and alpha == 1.0 :
                print(f"Converged after {i+1} iterations.")
                break
        else: # Executed if the loop completes without 'break' (max_iters reached)
            print("Reached maximum iterations.")

        return X, U, current_total_cost, cost_history

if __name__ == '__main__':
    # --- Example: Simple 1D system: x_dot = u (double integrator like) ---
    # More like: x_{k+1} = x_k + u_k * dt (position)
    # Let state be [position, velocity], control be [acceleration]
    # x_{k+1} = x_k + v_k * dt
    # v_{k+1} = v_k + a_k * dt

    STATE_DIM = 2
    CONTROL_DIM = 1
    HORIZON = 50
    DT = 0.1

    def simple_dynamics(state, control):
        x, v = state
        a, = control # control is a 1-element array

        new_x = x + v * DT
        new_v = v + a * DT
        return np.array([new_x, new_v])

    # Cost function: reach target state [1, 0] with minimum control effort
    TARGET_STATE = np.array([1.0, 0.0])
    R_control_cost = 0.1 # Control cost weight
    Q_state_cost = 1.0   # State cost weight (running)
    Q_final_cost = 100.0 # Final state cost weight

    def simple_cost(state, control, k, is_final=False, get_derivatives=False):
        # For this example, derivatives are computed numerically by DDP's _get_dynamics_jacobians
        # and _compute_derivatives will need to be adapted or this function expanded
        # to provide analytical derivatives for the cost.

        cost = 0
        if is_final:
            err = state - TARGET_STATE
            cost = 0.5 * Q_final_cost * np.dot(err, err)

            if get_derivatives:
                lx = Q_final_cost * err
                lu = np.zeros(CONTROL_DIM) # No control at final step
                lxx = Q_final_cost * np.eye(STATE_DIM)
                luu = np.zeros((CONTROL_DIM, CONTROL_DIM))
                lux = np.zeros((CONTROL_DIM, STATE_DIM))
                return cost, lx, lu, lxx, luu, lux
            return cost, None, None, None, None, None # Match tuple size
        else:
            err = state - TARGET_STATE # Penalize deviation from target along the way too
            cost_state = 0.5 * Q_state_cost * np.dot(err, err)
            cost_control = 0.5 * R_control_cost * np.dot(control, control)
            cost = cost_state + cost_control

            if get_derivatives:
                lx = Q_state_cost * err
                lu = R_control_cost * control
                lxx = Q_state_cost * np.eye(STATE_DIM)
                luu = R_control_cost * np.eye(CONTROL_DIM)
                lux = np.zeros((CONTROL_DIM, STATE_DIM))
                return cost, lx, lu, lxx, luu, lux
            return cost, None, None, None, None, None


    # Initialize iLQR Solver
    ilqr_solver = iLQRSolver(dynamics_fn=simple_dynamics,
                             cost_fn=simple_cost,
                             state_dim=STATE_DIM,
                             control_dim=CONTROL_DIM,
                             horizon=HORIZON)

    # Initial state and control guess
    x0 = np.array([0.0, 0.0]) # Start at origin, zero velocity
    U_initial = np.zeros((HORIZON, CONTROL_DIM)) # Initial guess of zero controls

    # Slightly perturb initial controls to avoid pure zero gradients if possible
    # U_initial = np.random.randn(HORIZON, CONTROL_DIM) * 0.01


    print("Running iLQR for simple 1D system...")
    X_opt, U_opt, final_cost, cost_history = ilqr_solver.run(x0, U_initial, max_iters=50, tol=1e-5) # Store cost_history

    print("\n--- iLQR Results ---")
    print(f"Final cost: {final_cost}")
    print(f"Initial state: {X_opt[0]}")
    print(f"Final state: {X_opt[-1]}")
    print(f"Target state: {TARGET_STATE}")

    # Plotting (optional, if matplotlib is available and display works)
    try:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        time = np.arange(HORIZON + 1) * DT

        # Plot position and velocity
        axs[0].plot(time, X_opt[:, 0], label='Position (x)')
        axs[0].plot(time, X_opt[:, 1], label='Velocity (v)')
        axs[0].axhline(TARGET_STATE[0], color='r', linestyle='--', label='Target Position')
        axs[0].axhline(TARGET_STATE[1], color='g', linestyle='--', label='Target Velocity')
        axs[0].set_ylabel('State')
        axs[0].legend()
        axs[0].grid(True)

        # Plot controls
        time_u = np.arange(HORIZON) * DT
        axs[1].plot(time_u, U_opt[:, 0], label='Control (acceleration)')
        axs[1].set_ylabel('Control')
        axs[1].legend()
        axs[1].grid(True)

        # Plot end-effector error (position error)
        position_error = X_opt[:, 0] - TARGET_STATE[0]
        axs[2].plot(time, position_error, label='Position Error')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Error')
        axs[2].legend()
        axs[2].grid(True)

        plt.suptitle('DDP Trajectory Optimization for Simple System')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Try to show plot, but be mindful of environment
        # plt.show(block=False) # Non-blocking
        # plt.pause(1) # Keep it open for a second
        print("\nPlot generated. If it doesn't display automatically, you may need to run in a GUI environment.")
        # To save:
        # plt.savefig("ddp_simple_system_results.png")
        # print("Plot saved to ddp_simple_system_results.png")

        # Hack to keep plot open
        try:
            while plt.fignum_exists(fig.number):
                plt.pause(0.1)
        except Exception:
            pass # Figure closed or other issue

    except ImportError:
        print("Matplotlib not found. Skipping plotting.")
    except Exception as e:
        print(f"Error during plotting: {e}. Skipping plotting.")

    print("iLQR example with manual/numerical derivatives finished.") # Changed DDP to iLQR

    # --- CasADi Example for the same 1D system ---
    print("\n--- Running iLQR for simple 1D system with CasADi derivatives ---")
    try:
        import casadi as ca

        # Define symbolic variables
        x_sym = ca.SX.sym('x', STATE_DIM)  # State [pos, vel]
        u_sym = ca.SX.sym('u', CONTROL_DIM)  # Control [accel]

        # Symbolic dynamics
        x_next_sym = ca.vertcat(
            x_sym[0] + x_sym[1] * DT,  # pos_next = pos + vel * DT
            x_sym[1] + u_sym[0] * DT   # vel_next = vel + accel * DT
        )
        f_sym = ca.Function('f_simple_sym', [x_sym, u_sym], [x_next_sym])

        # Symbolic Jacobians of dynamics
        fx_sym = ca.jacobian(x_next_sym, x_sym)
        fu_sym = ca.jacobian(x_next_sym, u_sym)
        f_x_func = ca.Function('f_x_simple', [x_sym, u_sym], [fx_sym])
        f_u_func = ca.Function('f_u_simple', [x_sym, u_sym], [fu_sym])

        # Symbolic cost function (running cost)
        err_sym_running = x_sym - TARGET_STATE
        l_running_sym = 0.5 * Q_state_cost * ca.dot(err_sym_running, err_sym_running) + \
                        0.5 * R_control_cost * ca.dot(u_sym, u_sym)

        # Symbolic cost function (terminal cost)
        err_sym_final = x_sym - TARGET_STATE
        l_final_sym = 0.5 * Q_final_cost * ca.dot(err_sym_final, err_sym_final)

        # Create callable functions for cost and its derivatives
        # Running cost derivatives
        l_x_sym = ca.gradient(l_running_sym, x_sym)
        l_u_sym = ca.gradient(l_running_sym, u_sym)
        l_xx_sym = ca.hessian(l_running_sym, x_sym)[0]
        l_uu_sym = ca.hessian(l_running_sym, u_sym)[0]
        l_ux_sym = ca.jacobian(l_u_sym, x_sym) # Hessian of l wrt u, then x (cross term)

        cost_running_func = ca.Function('l_running', [x_sym, u_sym], [l_running_sym, l_x_sym, l_u_sym, l_xx_sym, l_uu_sym, l_ux_sym])

        # Terminal cost derivatives
        l_final_x_sym = ca.gradient(l_final_sym, x_sym)
        l_final_xx_sym = ca.hessian(l_final_sym, x_sym)[0]
        # For terminal cost, lu, luu, lux are zero as there's no control input
        cost_final_func = ca.Function('l_final', [x_sym], [l_final_sym, l_final_x_sym, l_final_xx_sym])


        # Wrapper for the cost function to be used by the iLQRSolver
        def casadi_cost_wrapper(state, control, k, is_final, get_derivatives):
            if not get_derivatives: # DDP's forward pass only needs scalar cost
                if is_final:
                    return cost_final_func(state)[0].full().item(), None, None, None, None, None
                else:
                    return cost_running_func(state, control)[0].full().item(), None, None, None, None, None

            if is_final:
                cost, lx, lxx = cost_final_func(state)
                # Need to return all 6 derivative components, some will be zero
                lu = np.zeros(CONTROL_DIM)
                luu = np.zeros((CONTROL_DIM, CONTROL_DIM))
                lux = np.zeros((CONTROL_DIM, STATE_DIM))
                return cost.full().item(), lx.full().flatten(), lu, lxx.full(), luu, lux
            else:
                cost, lx, lu, lxx, luu, lux = cost_running_func(state, control)
                return cost.full().item(), lx.full().flatten(), lu.full().flatten(), \
                       lxx.full(), luu.full(), lux.full()

        # Wrapper for dynamics to be used by iLQRSolver if it still uses it for rollouts
        # The current iLQRSolver uses self.dynamics_fn for rollouts.
        # If we modify iLQRSolver to take fx_func, fu_func for derivative calculations,
        # it still needs a way to simulate.
        # This numerical_dynamics_fn can still be the original one.
        numerical_dynamics_fn = simple_dynamics # Original numerical one for rollouts

        # --- Option A: Modify iLQRSolver to accept precomputed derivative functions ---
        # This would be the cleaner way. For this example, we'll simulate this by
        # overriding _get_dynamics_jacobians if the solver were an instance variable
        # or by creating a modified solver.
        # For now, let's assume the solver is modified to use these if provided.
        # (This part will be more fully addressed in the next plan step: Refactor Solvers)

        # Create a temporary iLQRSolver that uses CasADi derivatives for this example
        # This is a bit of a hack for the example without modifying the main class yet.
        class iLQRSolverWithCasADi(iLQRSolver):
            def _get_dynamics_jacobians(self, x, u):
                # Call the CasADi-generated functions
                fx_val = f_x_func(x, u).full()
                fu_val = f_u_func(x, u).full()
                return fx_val, fu_val

        ilqr_solver_casadi = iLQRSolverWithCasADi(
            dynamics_fn=numerical_dynamics_fn, # Still needed for rollouts
            cost_fn=casadi_cost_wrapper,       # Cost function now uses CasADi derivatives
            state_dim=STATE_DIM,
            control_dim=CONTROL_DIM,
            horizon=HORIZON
        )

        print("Running iLQR (with CasADi derivatives) for simple 1D system...")
        X_opt_casadi, U_opt_casadi, final_cost_casadi, cost_history_casadi = ilqr_solver_casadi.run(
            x0, U_initial, max_iters=50, tol=1e-5
        )

        print("\n--- iLQR (CasADi) Results ---")
        print(f"Final cost: {final_cost_casadi}")
        print(f"Initial state: {X_opt_casadi[0]}")
        print(f"Final state: {X_opt_casadi[-1]}")

        # (Optional) Add plotting for CasADi results similar to the above example

    except ImportError:
        print("CasADi not found. Skipping CasADi example.")
    except Exception as e:
        print(f"Error during CasADi example: {e}")

class FullDDPSolver(iLQRSolver):
    def __init__(self, dynamics_fn, cost_fn, state_dim, control_dim, horizon,
                 dynamics_hessians_fn=None): # Can accept a function to provide Hessians
        """
        Full Differential Dynamic Programming (DDP) Solver (Conceptual Sketch).
        Inherits from iLQRSolver and overrides parts of the backward pass
        to include second-order derivatives of the dynamics.

        Args:
            dynamics_fn, cost_fn, state_dim, control_dim, horizon: Same as iLQRSolver.
            dynamics_hessians_fn (callable, optional):
                A function `fxx, fuu, fux = func(x, u)` that returns Hessians of dynamics.
                If None, these terms are conceptually ignored (falling back to iLQR-like behavior
                for those specific terms, or requiring numerical computation not implemented here).
        """
        super().__init__(dynamics_fn, cost_fn, state_dim, control_dim, horizon)
        self.dynamics_hessians_fn = dynamics_hessians_fn
        if self.dynamics_hessians_fn is None:
            print("Warning: FullDDPSolver initialized without dynamics_hessians_fn. "
                  "Second-order dynamics terms will be missing, behaving like iLQR for those parts.")

    def _get_dynamics_hessians(self, x, u):
        """
        Placeholder/Conceptual function to get Hessians of dynamics f(x,u).
        f_xx = ∂²f/∂x² (a tensor)
        f_uu = ∂²f/∂u² (a tensor)
        f_ux = ∂²f/∂u∂x (a tensor)

        In a real implementation, these would be computed analytically or numerically.
        This sketch assumes they are provided or are zero if not.
        """
        if self.dynamics_hessians_fn:
            try:
                f_xx, f_uu, f_ux = self.dynamics_hessians_fn(x, u)
                return f_xx, f_uu, f_ux
            except Exception as e:
                print(f"Error calling dynamics_hessians_fn: {e}. Returning zero Hessians.")
                # Fallback to zero Hessians if the provided function fails

        # Placeholder: return zero Hessians if no function is provided.
        # This makes the DDP equations reduce to iLQR equations for these terms.
        # Dimensions:
        # f_xx: (state_dim, state_dim, state_dim) -> ∂f_i / ∂x_j ∂x_k
        # f_uu: (state_dim, control_dim, control_dim) -> ∂f_i / ∂u_j ∂u_k
        # f_ux: (state_dim, control_dim, state_dim) -> ∂f_i / ∂u_j ∂x_k
        # For simplicity, this sketch doesn't define the exact tensor shapes if zero.
        # A full implementation would need careful handling of these tensor shapes and contractions.
        print("Warning: _get_dynamics_hessians called but no function provided or it failed. Returning zero Hessians.")
        f_xx = np.zeros((self.state_dim, self.state_dim, self.state_dim))
        f_uu = np.zeros((self.state_dim, self.control_dim, self.control_dim))
        f_ux = np.zeros((self.state_dim, self.control_dim, self.state_dim))
        return f_xx, f_uu, f_ux


    def backward_pass(self, X, U, derivatives):
        """
        Performs the backward pass for Full DDP.
        Overrides iLQRSolver's backward_pass to include second-order dynamics terms.
        """
        V_x = derivatives['lx_N']
        V_xx = derivatives['lxx_N']

        k_feedforward_terms = []
        K_feedback_terms = []

        current_regularization = self.reg_factor

        for k in range(self.N - 1, -1, -1):
            lx = derivatives['lx'][k]
            lu = derivatives['lu'][k]
            lxx = derivatives['lxx'][k]
            luu = derivatives['luu'][k]
            lux = derivatives['lux'][k]
            fx = derivatives['fx'][k] # Jacobian df/dx
            fu = derivatives['fu'][k] # Jacobian df/du

            # Q-function expansion terms
            Q_x = lx + fx.T @ V_x
            Q_u = lu + fu.T @ V_x

            Q_xx = lxx + fx.T @ V_xx @ fx
            Q_uu = luu + fu.T @ V_xx @ fu
            Q_ux = lux + fu.T @ V_xx @ fx

            # --- Full DDP modification: Add terms with second-order dynamics derivatives ---
            # These require Hessians of dynamics: f_xx, f_uu, f_ux
            # And involve tensor contractions with V_x (gradient of Value function from next step)
            # For this sketch, we call a placeholder _get_dynamics_hessians.
            # A robust implementation would require careful numerical or analytical calculation of these.

            # Conceptually, if f_xx, f_uu, f_ux were available (e.g. state_dim x state_dim x state_dim tensors):
            f_xx_k, f_uu_k, f_ux_k = self._get_dynamics_hessians(X[k], U[k])

            # The contraction V_x' * f_xx is sum_i V_x[i] * f_xx[i,:,:]
            # where f_xx[i,:,:] is the Hessian of the i-th component of f.
            # This results in a matrix of size state_dim x state_dim.
            # Similar contractions for f_uu and f_ux.
            # Note: np.einsum is a powerful tool for such contractions.
            # Example (conceptual, exact einsum string depends on tensor dimension ordering):
            if self.dynamics_hessians_fn: # Only add if Hessians are meaningfully computed
                # Q_xx_ddp_term = np.einsum('i,ijk->jk', V_x, f_xx_k) # V_x is (state_dim), f_xx_k is (state_dim, state_dim, state_dim)
                # Q_uu_ddp_term = np.einsum('i,ijk->jk', V_x, f_uu_k) # f_uu_k is (state_dim, control_dim, control_dim)
                # Q_ux_ddp_term = np.einsum('i,ijk->jk', V_x, f_ux_k) # f_ux_k is (state_dim, control_dim, state_dim)

                # Simplified placeholder for sketch - actual tensor math is more involved.
                # This assumes f_**_k are already contracted with V_x if not zero.
                # For a real implementation, one would pass V_x to _get_dynamics_hessians
                # or perform the contraction here carefully.
                # For this sketch, we assume _get_dynamics_hessians might return already-contracted terms or zeros.
                # If they return raw Hessians, the einsum above is needed.
                # Let's assume for the sketch, these are *additional* components.
                # This part is highly conceptual in this sketch.

                # A more direct way for the sketch, assuming f_**_k are raw Hessians:
                # Q_xx += np.tensordot(V_x, f_xx_k, axes=1) # if f_xx_k[i,j,k] = d(f_i)/dx_j dx_k
                # Q_uu += np.tensordot(V_x, f_uu_k, axes=1)
                # Q_ux += np.tensordot(V_x, f_ux_k, axes=1) # Check axes for f_ux
                 pass # Not implementing the actual contraction due to complexity of general tensor setup


            Q_uu_reg = Q_uu + np.eye(self.control_dim) * current_regularization

            max_reg_increases = 10
            for _ in range(max_reg_increases):
                try:
                    np.linalg.cholesky(Q_uu_reg)
                    break
                except np.linalg.LinAlgError:
                    current_regularization = max(self.reg_min, current_regularization * self.reg_factor)
                    current_regularization = min(current_regularization, self.reg_max)
                    Q_uu_reg = Q_uu + np.eye(self.control_dim) * current_regularization
                    # print(f"    (FullDDP) Warning: Quu not positive definite at k={k}. Reg: {current_regularization:.2e}")
            else:
                print(f"    (FullDDP) ERROR: Quu not positive definite at k={k}. Max reg: {current_regularization:.2e}. Aborting.")
                return None, None, False

            try:
                k_k = np.linalg.solve(Q_uu_reg, -Q_u)
                K_k = np.linalg.solve(Q_uu_reg, -Q_ux)
            except np.linalg.LinAlgError:
                 print(f"    (FullDDP) ERROR: Solving for k_k, K_k failed at k={k}. Aborting.")
                 return None, None, False

            k_feedforward_terms.insert(0, k_k)
            K_feedback_terms.insert(0, K_k)

            V_x = Q_x + K_k.T @ Q_uu_reg @ k_k + K_k.T @ Q_u + Q_ux.T @ k_k
            V_xx = Q_xx + K_k.T @ Q_uu_reg @ K_k + K_k.T @ Q_ux + Q_ux.T @ K_k
            V_xx = 0.5 * (V_xx + V_xx.T)

        return k_feedforward_terms, K_feedback_terms, True
