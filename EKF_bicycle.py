import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Time step
dt = 0.1

# Bicycle parameters
L = 2.0  # Bicycle wheelbase (distance between the two axles)
max_steering_angle = np.pi / 3  # Maximum steering angle (45 degrees)
max_velocity = 5.0  # Maximum velocity

# Noise parameters (for process and measurement noise)
process_noise_std = np.array([0.1, 0.1, 0.05, 0.1])  # Process noise for [x, y, theta, v]
measurement_noise_std = np.array([0.2, 0.2])  # Measurement noise for [x, y]

# Reference trajectory (sine wave path)
def reference_trajectory(timesteps):
    x_ref = np.linspace(0, 20, timesteps)
    y_ref = 2 * np.sin( x_ref)  # Sine wave function
    return np.vstack((x_ref, y_ref))

# Bicycle model dynamics
def bicycle_model(state, control):
    x, y, theta, v = state
    delta = control[0]  # Steering angle
    v = control[1]  # Velocity
    
    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt
    theta_next = theta + (v / L) * np.tan(delta) * dt
    
    return np.array([x_next, y_next, theta_next, v])

# Jacobian matrix of the state transition function (for linearization in EKF)
def jacobian_f(state, control):
    x, y, theta, v = state
    delta = control[0]
    
    # Jacobian of the state transition model
    F = np.eye(4)
    F[0, 2] = -v * np.sin(theta) * dt
    F[0, 3] = np.cos(theta) * dt
    F[1, 2] = v * np.cos(theta) * dt
    F[1, 3] = np.sin(theta) * dt
    F[2, 3] = (1 / L) * np.tan(delta) * dt
    
    return F

# Jacobian matrix of the measurement function (identity matrix in this case)
def jacobian_h():
    H = np.eye(2, 4)
    return H

# Measurement model (extracts only x and y from the state)
def measurement_model(state):
    return state[:2]

# EKF Prediction step
def ekf_predict(state, control, P, Q):
    # State transition
    state_pred = bicycle_model(state, control)
    
    # Jacobian of the state transition model
    F = jacobian_f(state, control)
    
    # Covariance prediction
    P_pred = F @ P @ F.T + Q
    
    return state_pred, P_pred

# EKF Update step
def ekf_update(state_pred, P_pred, measurement, R):
    H = jacobian_h()
    
    # Measurement prediction
    z_pred = measurement_model(state_pred)
    
    # Innovation covariance
    S = H @ P_pred @ H.T + R
    
    # Kalman Gain
    K = P_pred @ H.T @ np.linalg.inv(S)
    
    # Update state estimate
    state_updated = state_pred + K @ (measurement - z_pred)
    
    # Update covariance estimate
    P_updated = (np.eye(len(state_pred)) - K @ H) @ P_pred
    
    return state_updated, P_updated

# Cost function for steering and velocity optimization (minimize error between predicted state and reference trajectory)
def control_cost_function(control, *args):
    state, ref_horizon = args
    total_error = 0
    v = control[1]  # Velocity
    delta = control[0]  # Steering angle
    
    for ref in ref_horizon.T:
        state = bicycle_model(state, [delta, v])
        error = np.linalg.norm(state[:2] - ref)  # Distance error
        total_error += error
    
    return total_error

# Simulate EKF with optimized steering and velocity control
def simulate_ekf_with_steering_velocity_optimization():
    # Initial state [x, y, theta, v]
    state = np.array([0, 0, 0, 1.0])
    
    # Initial state covariance
    P = np.eye(4) * 0.1
    
    # Process noise covariance
    Q = np.diag(process_noise_std**2)
    
    # Measurement noise covariance
    R = np.diag(measurement_noise_std**2)
    
    # Reference trajectory
    timesteps = 200
    ref_traj = reference_trajectory(timesteps)
    
    # Initial control input [steering angle, velocity]
    control = np.array([0.0, 1.0])
    
    # Trajectories for visualization
    estimated_trajectory = []
    true_trajectory = []
    measurements = []
    
    for t in range(timesteps - 10):
        # Get current reference trajectory segment for control optimization
        ref_horizon = ref_traj[:, t:t + 10]
        
        # Optimize both steering and velocity using MPC
        control_result = minimize(control_cost_function, control, args=(state, ref_horizon),
                                  bounds=[(-max_steering_angle, max_steering_angle), (0.1, max_velocity)])
        control = control_result.x
        
        # Simulate true dynamics (with process noise)
        true_state = bicycle_model(state, control) + np.random.normal(0, process_noise_std, size=state.shape)
        true_trajectory.append(true_state[:2])
        
        # Simulate noisy measurement
        measurement = measurement_model(true_state) + np.random.normal(0, measurement_noise_std)
        measurements.append(measurement)
        
        # EKF Prediction step
        state_pred, P_pred = ekf_predict(state, control, P, Q)
        
        # EKF Update step
        state, P = ekf_update(state_pred, P_pred, measurement, R)
        
        estimated_trajectory.append(state[:2])
    
    return np.array(estimated_trajectory), np.array(true_trajectory), np.array(measurements), ref_traj

# Plotting the results
def plot_trajectory(estimated_trajectory, true_trajectory, measurements, ref_traj):
    plt.plot(ref_traj[0, :], ref_traj[1, :], 'g--', label='Reference Trajectory (Sine Wave)')
    plt.plot(true_trajectory[:, 0], true_trajectory[:, 1], 'b-', label='True Trajectory')
    plt.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], 'r-', label='Estimated Trajectory (EKF)')
    plt.scatter(measurements[:, 0], measurements[:, 1], color='black', marker='x', label='Noisy Measurements')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Bicycle Path Tracking with EKF and Optimized Control Inputs')
    plt.grid()
    plt.show()

# Main function to run the simulation and plot results
if __name__ == "__main__":
    estimated_trajectory, true_trajectory, measurements, ref_traj = simulate_ekf_with_steering_velocity_optimization()
    plot_trajectory(estimated_trajectory, true_trajectory, measurements, ref_traj)
