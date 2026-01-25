# AGENTS.md

## Project Overview
This project implements the Iterative Linear Quadratic Regulator (iLQR) algorithm for optimizing the motion of a 2-link planar robotic arm. The goal is to drive the arm's end-effector to a target position while minimizing control effort and satisfying defined objectives.

## Key Files & Structure
- **`src/robot_env.py`**: Defines the 2-link arm simulation (kinematics, dynamics) and visualization (`RobotVisualizer`).
- **`src/ddp.py`**: Contains the `iLQRSolver` class which implements the iLQR optimization algorithm.
- **`src/main.py`**: The main entry point. Sets up the environment, defines the cost function, runs the iLQR optimization, and visualizes the results.
- **`tests/`**: Directory containing unit tests.

## Workflow & Commands
- **Testing**: Always verify changes by running the test suite:
  ```bash
  python -m pytest
  ```
- **Running the Application**: To run the main simulation and visualization:
  ```bash
  python src/main.py
  ```

## Coding Guidelines
- **Style**: Follow the existing coding style and conventions found in the codebase.
- **Consistency**: Ensure variable naming and structure remain consistent with current files (e.g., snake_case for functions/variables, PascalCase for classes).
- **Context**: Minimize context consumption by focusing on the specific files relevant to the task.

## Restrictions
- None at this time.
