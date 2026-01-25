import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

class TwoLinkArm:
    def __init__(self, link_lengths=[1.0, 1.0], initial_angles_rad=[0.0, 0.0]):
        """
        Represents a simple 2-link planar robotic arm.

        The state of the arm for DDP is [theta1, theta2, omega1, omega2], where:
        - theta1: Angle of the first link (from horizontal).
        - theta2: Angle of the second link (relative to the first link).
        - omega1: Angular velocity of the first link.
        - omega2: Angular velocity of the second link.

        Controls for DDP are [alpha1, alpha2] (angular accelerations).

        Args:
            link_lengths (list of float): Lengths of the two links [l1, l2].
            initial_angles_rad (list of float): Initial joint angles [theta1, theta2] in radians.
        """
        self.link_lengths = np.array(link_lengths, dtype=float)
        # self.current_angles_rad stores only the angles for internal FK use if needed,
        # but DDP state 'x' will carry the full [theta1, theta2, omega1, omega2].
        self.current_angles_rad = np.array(initial_angles_rad, dtype=float)

        self.max_angular_velocity = np.pi / 2  # rad/s (increased from pi/4 for potentially faster moves)
        self.dt = 0.1  # Time step for simulation (seconds)

        self.state_dim = 4  # [theta1, theta2, omega1, omega2]
        self.control_dim = 2 # [alpha1, alpha2] - joint angular accelerations

    def forward_kinematics(self, joint_angles_rad):
        """
        Computes the positions of the base, elbow, and end-effector.

        Args:
            joint_angles_rad (np.array): Joint angles [theta1, theta2] in radians.
                                      theta1 is angle of link1 w.r.t positive x-axis.
                                      theta2 is angle of link2 w.r.t the orientation of link1.
        Returns:
            tuple: (p_base, p_elbow, p_end_effector)
                p_base (np.array): Position of the base (always [0,0]).
                p_elbow (np.array): Position of the elbow [x, y].
                p_end_effector (np.array): Position of the end-effector [x, y].
        """
        l1, l2 = self.link_lengths
        theta1, theta2 = joint_angles_rad

        # Position of the first joint (elbow)
        x_elbow = l1 * np.cos(theta1)
        y_elbow = l1 * np.sin(theta1)

        # Position of the second joint (end-effector)
        # Angle of link2 in world frame is theta1 + theta2
        x_ee = x_elbow + l2 * np.cos(theta1 + theta2)
        y_ee = y_elbow + l2 * np.sin(theta1 + theta2)

        return np.array([0.0, 0.0]), np.array([x_elbow, y_elbow]), np.array([x_ee, y_ee])

    def set_angles(self, angles_rad):
        """Sets the internal current angles of the arm."""
        self.current_angles_rad = np.array(angles_rad, dtype=float)

    def dynamics(self, state, control_accelerations):
        """
        Computes the next state of the arm given the current state and control inputs (accelerations).
        Uses simple Euler integration.

        Args:
            state (np.array): Current state [theta1, theta2, omega1, omega2] in radians and rad/s.
            control_accelerations (np.array): Control inputs [alpha1, alpha2] (joint angular accelerations in rad/s^2).

        Returns:
            np.array: Next state [theta1_new, theta2_new, omega1_new, omega2_new].
        """
        theta1, theta2, omega1, omega2 = state
        alpha1, alpha2 = control_accelerations # These are angular accelerations

        # Integrate accelerations to get new velocities
        new_omega1 = omega1 + alpha1 * self.dt
        new_omega2 = omega2 + alpha2 * self.dt

        # Apply velocity limits
        new_omega1 = np.clip(new_omega1, -self.max_angular_velocity, self.max_angular_velocity)
        new_omega2 = np.clip(new_omega2, -self.max_angular_velocity, self.max_angular_velocity)

        # Integrate velocities to get new angles
        new_theta1 = theta1 + new_omega1 * self.dt
        new_theta2 = theta2 + new_omega2 * self.dt

        # Optional: Normalize angles to a specific range, e.g., [-pi, pi] or [0, 2pi]
        # This can help prevent issues with very large angle values if the simulation runs for long.
        # new_theta1 = (new_theta1 + np.pi) % (2 * np.pi) - np.pi
        # new_theta2 = (new_theta2 + np.pi) % (2 * np.pi) - np.pi

        return np.array([new_theta1, new_theta2, new_omega1, new_omega2])

    def reset(self, initial_angles_rad=[0.0, 0.0], initial_omegas_rad_s=[0.0, 0.0]):
        """
        Resets the arm to a specified initial configuration.

        Args:
            initial_angles_rad (list or np.array): Initial joint angles [theta1, theta2] in radians.
            initial_omegas_rad_s (list or np.array): Initial joint velocities [omega1, omega2] in rad/s.

        Returns:
            np.array: The initial state vector [theta1, theta2, omega1, omega2].
        """
        self.current_angles_rad = np.array(initial_angles_rad, dtype=float)
        initial_state_vector = np.concatenate((self.current_angles_rad, np.array(initial_omegas_rad_s, dtype=float)))
        return initial_state_vector


class RobotVisualizer:
    """Handles the Matplotlib-based visualization of the TwoLinkArm."""
    def __init__(self, arm_instance, target_pos_xy=None):
        """
        Initializes the visualizer.

        Args:
            arm_instance (TwoLinkArm): An instance of the TwoLinkArm to visualize.
            target_pos_xy (np.array, optional): Target [x,y] position to display as a marker.
        """
        self.arm = arm_instance
        self.target_pos = target_pos_xy

        self.fig, self.ax = plt.subplots(figsize=(7,7)) # Create a figure and an axes.
        self.ax.set_aspect('equal') # Ensure aspect ratio is equal to avoid distortion.

        # Calculate plot limits based on arm's maximum reach
        max_reach = sum(self.arm.link_lengths)
        self.ax.set_xlim(-max_reach - 0.5, max_reach + 0.5)
        self.ax.set_ylim(-max_reach - 0.5, max_reach + 0.5)

        # Create line objects for the arm links and points for joints/end-effector
        self.link1_line, = self.ax.plot([], [], 'b-', lw=3, label='Link 1') # Blue, solid, thicker line
        self.link2_line, = self.ax.plot([], [], 'r-', lw=3, label='Link 2') # Red, solid, thicker line
        self.joint0_dot, = self.ax.plot([], [], 'ko', ms=8, label='Base')    # Black circle marker
        self.joint1_dot, = self.ax.plot([], [], 'ko', ms=8)                  # Elbow joint
        self.end_effector_dot, = self.ax.plot([], [], 'go', ms=8, label='End-Effector') # Green circle for EE

        # Add target marker if specified
        if self.target_pos is not None:
            self.target_marker = patches.Circle(self.target_pos, radius=0.05, fc='gray', alpha=0.7, label='Target')
            self.ax.add_patch(self.target_marker)

        self.ax.set_xlabel("X position (m)")
        self.ax.set_ylabel("Y position (m)")
        self.title = self.ax.set_title("2-Link Robotic Arm Simulation")
        self.ax.legend(loc='upper right')
        self.ax.grid(True)


    def update_plot(self, joint_angles_rad):
        """
        Updates the arm's position on the plot given new joint angles.

        Args:
            joint_angles_rad (np.array): Current joint angles [theta1, theta2] in radians.

        Returns:
            tuple: A tuple of updated Matplotlib artists (lines and points).
        """
        p_base, p_elbow, p_ee = self.arm.forward_kinematics(joint_angles_rad)

        self.link1_line.set_data([p_base[0], p_elbow[0]], [p_base[1], p_elbow[1]])
        self.link2_line.set_data([p_elbow[0], p_ee[0]], [p_elbow[1], p_ee[1]])
        self.joint0_dot.set_data(p_base[0], p_base[1])
        self.joint1_dot.set_data(p_elbow[0], p_elbow[1])
        self.end_effector_dot.set_data(p_ee[0], p_ee[1])

        return (self.link1_line, self.link2_line,
                self.joint0_dot, self.joint1_dot, self.end_effector_dot)

    def animate_trajectory(self, trajectory_states_list, frame_interval_ms=100):
        """
        Animates the robot arm's movement given a trajectory of states.

        Args:
            trajectory_states_list (list or np.array): A sequence of states.
                Each state is expected to be [theta1, theta2, omega1, omega2].
                Only theta1 and theta2 are used for plotting the arm's configuration.
            frame_interval_ms (int): Time interval between frames in milliseconds.

        Returns:
            matplotlib.animation.FuncAnimation: The animation object.
        """
        # anim_init function for FuncAnimation: plots the background of each frame.
        def anim_init_func():
            # Initialize with the first frame's angles from the trajectory
            initial_joint_angles = trajectory_states_list[0][:2] # Extract [theta1, theta2]
            return self.update_plot(initial_joint_angles)

        # anim_update function for FuncAnimation: called for each frame.
        def anim_update_func(frame_index):
            current_joint_angles = trajectory_states_list[frame_index][:2] # Extract [theta1, theta2]
            updated_artists = self.update_plot(current_joint_angles)
            self.title.set_text(f"Step {frame_index}/{len(trajectory_states_list)-1}")
            return updated_artists

        # Create the animation
        # `blit=True` means only re-draw the parts that have changed for efficiency.
        animation_object = FuncAnimation(self.fig, anim_update_func,
                                         frames=len(trajectory_states_list),
                                         init_func=anim_init_func,
                                         blit=True,
                                         interval=frame_interval_ms,
                                         repeat=False) # Do not repeat animation

        if plt.get_backend() != 'agg': # pragma: no cover
             plt.show(block=False) # Show the plot (non-blocking for script flow)

        return animation_object

if __name__ == '__main__':
    # --- Test the TwoLinkArm and Visualizer ---
    arm = TwoLinkArm(link_lengths=[1.0, 0.8], initial_angles=[np.pi/4, np.pi/4])
    target_position = np.array([1.0, 1.0]) # Example target

    print("Initial Arm State (angles, omegas):", arm.reset(initial_angles=[np.pi/4, np.pi/4]))

    # Test forward kinematics
    p0, p1, p2 = arm.forward_kinematics()
    print("Joint positions:")
    print("P0 (Base):", p0)
    print("P1 (Elbow):", p1)
    print("P2 (End-effector):", p2)

    # Test dynamics: Apply some control
    initial_state = arm.reset(initial_angles=[0,0])
    print(f"Initial state: {initial_state}")

    # Example control: accelerate first joint, decelerate second
    controls = [np.array([0.1, -0.05])] * 20 + [np.array([-0.05, 0.1])] * 30

    trajectory = [initial_state]
    current_state = initial_state
    for i in range(50):
        control_action = controls[i]
        next_state = arm.dynamics(current_state, control_action)
        trajectory.append(next_state)
        current_state = next_state
        # arm.set_angles(current_state[:2]) # Not needed if visualizer takes full state

    print(f"Final state after 50 steps: {current_state}")
    print(f"End effector final position: {arm.forward_kinematics(current_state[:2])[2]}")

    # Visualize the trajectory
    vis = RobotVisualizer(arm, target_pos=target_position)

    # To show initial position before animation:
    vis.update_plot(initial_state[:2])
    plt.pause(1) # Show initial pose for 1 second

    # Animate the generated trajectory
    # The FuncAnimation will run the simulation and display it.
    # Note: Matplotlib animations might behave differently in different environments (e.g., scripts vs notebooks)
    # If the plot doesn't show or closes immediately, you might need plt.show(block=True) or ensure your environment supports GUI event loops.
    print("Starting animation...")
    animation = vis.animate_trajectory(trajectory, interval=int(arm.dt * 1000))
    # For script execution, plt.show() might be needed here if not called by FuncAnimation internally
    # For some backends, the animation object must be kept in scope

    # If you want to save the animation:
    # animation.save('arm_trajectory.gif', writer='imagemagick', fps=1/arm.dt)
    # print("Animation saved to arm_trajectory.gif")

    print("Visualization complete. Close the plot window to exit.")
    # plt.show() # This will block until the window is closed. FuncAnimation usually handles this.
    # If FuncAnimation doesn't block, uncomment the line above.
    # In many interactive environments, the window stays open. If not, uncomment:
    # plt.show(block=True)

    # Hack to keep plot open in some environments if FuncAnimation doesn't block
    try:
        while plt.fignum_exists(vis.fig.number):
            plt.pause(0.1)
    except Exception as e:
        print(f"Plot window closed or error during wait: {e}")

    print("Script finished.")
