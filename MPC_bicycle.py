import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Bicycle parameters
L = 2.0     # Wheelbase
dt = 0.1     # Time step
v = 1 # Constant velocity

# Prediction and control horizons
f = 8    # Prediction horizon
v_control = 3 # Control horizon

# Weight matrices
W_state = np.eye(2)*1000000000 # Weight for state tracking (x, y)
W_control = 0.0000000000001   # Weight for control input (steering angle)

# Define the reference trajectory (sine wave)
def reference_trajectory(timesteps):
    x_ref = np.linspace(0, 20, timesteps)
    y_ref =  np.sin(x_ref)  # Sine wave function
    return np.vstack((x_ref, y_ref))

# Bicycle model dynamics
def bicycle_model(state, control_input):
    x, y, theta = state
    delta = control_input

    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt
    theta_next = theta + (v / L) * np.tan(delta) * dt

    return np.array([x_next, y_next, theta_next])

# Objective function to minimize
def mpc_objective(u, *args):
    state, ref_traj = args
    cost = 0.0

    for i in range(f):  # Prediction horizon
        # Apply control input and predict the next state
        state = bicycle_model(state, u[i])

        # Calculate error between predicted state and reference
        error = state[:2] - ref_traj[:, i]  # Error in (x, y)
        cost += np.dot(error.T, np.dot(W_state, error))  # State cost

        # Penalize control input (steering)
        cost += W_control * u[i]**2  # Control cost

    return cost

# MPC controller
def mpc_controller(current_state, ref_traj):
    # Initial guess for control inputs (steering angles)
    u0 = np.zeros(f)

    # Optimization constraints (steering limits)
    bounds = [(-np.pi/3, np.pi/3)] * f  # Steering angle limits

    # Solve the optimization problem
    result = minimize(mpc_objective, u0, args=(current_state, ref_traj), bounds=bounds)

    # Extract the first control input (steering angle) from the result
    return result.x[0]  # Only the first control input is used

# Simulate the bicycle and MPC controller
def simulate_mpc():
    # Initial state [x, y, theta]
    state = np.array([0, 0, 0])

    # Time steps for simulation
    timesteps = 250
    trajectory = [state]

    # Get the reference trajectory (sine wave)
    ref_traj = reference_trajectory(timesteps)

    for t in range(timesteps - f):
        # Get the current reference trajectory for the prediction horizon
        ref_horizon = ref_traj[:, t:t+f]

        # Get the optimal control input from MPC
        control_input = mpc_controller(state, ref_horizon)

        # Apply the control input to the bicycle model
        state = bicycle_model(state, control_input)

        # Store the new state
        trajectory.append(state)

    return np.array(trajectory), ref_traj

# Plot the results
def plot_trajectory(trajectory, ref_traj):
    plt.plot(ref_traj[0, :], ref_traj[1, :], label='Reference Trajectory (Sine Wave)', linestyle='--')
    plt.plot(trajectory[:, 0], trajectory[:, 1], label='MPC Bicycle Path')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Bicycle Path Following Using MPC')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function to run the simulation
if __name__ == "__main__":
    trajectory, ref_traj = simulate_mpc()
    plot_trajectory(trajectory, ref_traj)
