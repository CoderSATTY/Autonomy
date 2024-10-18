## Unpacking the Data
import pickle
import numpy as np
import matplotlib.pyplot as plt

with open(r'C:\Users\Hp\Downloads\data.pickle', 'rb') as f:
    data = pickle.load(f)

t = data['t']  # timestamps [s]

x_init  = data['x_init'] # initial x position [m]
y_init  = data['y_init'] # initial y position [m]
th_init = data['th_init'] # initial theta position [rad]

# input signal
v  = data['v']  # translational velocity input [m/s]
om = data['om']  # rotational velocity input [rad/s]

# bearing and range measurements, LIDAR constants
b = data['b']  # bearing to each landmarks center in the frame attached to the laser [rad]
r = data['r']  # range measurements [m]
l = data['l']  # x,y positions of landmarks [m]
d = data['d']  # distance between robot center and laser rangefinder [m]


## Testing the Unpacked Data
print(v.shape)
print(len(v))
print(d.shape)
print(d[0])

## Initializing Parameters
v_var = 0.01  # translation velocity variance
om_var = 0.01  # rotational velocity variance
r_var = 0.01   # range measurements variance (tuned)
b_var = 10     # bearing measurement variance (tuned)

Q_km = np.diag([v_var, om_var]) # input noise covariance
cov_y = np.diag([r_var, b_var])  # measurement noise covariance

x_est = np.zeros([len(v), 3])  # estimated states, x, y, and theta
P_est = np.zeros([len(v), 3, 3])  # state covariance matrices

x_est[0] = np.array([x_init, y_init, th_init]) # initial state
P_est[0] = np.diag([1, 1, 0.1]) # initial state covariance

print(Q_km.shape)
print(Q_km)
print(cov_y.shape)
print(cov_y)
print(P_est.shape)
print(P_est[0])


# Wraps angle to (-pi, pi] range
def wraptopi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


## Correction Step
def measurement_update(lk, rk, bk, P_check, x_check):
    x_k = x_check[0, 0]
    y_k = x_check[1, 0]
    theta_k = wraptopi(x_check[2].item()) if hasattr(x_check[2], 'item') else wraptopi(x_check[2])
    if lk.shape != (2,):
        raise ValueError(f"Landmark position 'lk' has incorrect shape: {lk.shape}, expected (2,)")
    x_l = lk[0]
    y_l = lk[1]

    d_x = x_l - (x_k + d * np.cos(theta_k))
    d_y = y_l - (y_k + d * np.sin(theta_k))

    r = np.sqrt(d_x**2 + d_y**2)
    phi = wraptopi(np.arctan2(d_y, d_x) - theta_k)

    # 1. Compute measurement Jacobian
    H_k = np.zeros((2, 3))
# Ensure scalar values using .item() to extract scalar from array
    H_k[0, 0] = (-d_x / r).item() if hasattr(-d_x / r, 'item') else -d_x / r
    H_k[0, 1] = (-d_y / r).item() if hasattr(-d_y / r, 'item') else -d_y / r
    H_k[0, 2] = (d * (d_x * np.sin(theta_k) - d_y * np.cos(theta_k)) / r).item() if hasattr(d * (d_x * np.sin(theta_k) - d_y * np.cos(theta_k)) / r, 'item') else d * (d_x * np.sin(theta_k) - d_y * np.cos(theta_k)) / r
    H_k[1, 0] = (d_y / r**2).item() if hasattr(d_y / r**2, 'item') else d_y / r**2
    H_k[1, 1] = (-d_x / r**2).item() if hasattr(-d_x / r**2, 'item') else -d_x / r**2
    H_k[1, 2] = (-1 - d * (d_y * np.sin(theta_k) + d_x * np.cos(theta_k)) / r**2).item() if hasattr(-1 - d * (d_y * np.sin(theta_k) + d_x * np.cos(theta_k)) / r**2, 'item') else -1 - d * (d_y * np.sin(theta_k) + d_x * np.cos(theta_k))


    M_k = np.identity(2)
    
    # Output and measurement vector
    y_out = np.vstack([r, wraptopi(phi)])
    y_mes = np.vstack([rk, wraptopi(bk)])

    # 2. Compute Kalman Gain
    S_k = H_k @ P_check @ H_k.T + M_k @ cov_y @ M_k.T
    K_k = P_check @ H_k.T @ np.linalg.inv(S_k)

    # 3. Correct predicted state (wrap the angles to [-pi,pi])
    x_check = x_check + K_k @ (y_mes - y_out)
    x_check[2] = wraptopi(x_check[2])

    # 4. Correct covariance
    P_check = (np.identity(3) - K_k @ H_k) @ P_check

    return x_check, P_check


## Prediction Step
P_check = P_est[0]
x_check = x_est[0, :].reshape(3, 1)

for k in range(1, len(t)):  # start at 1 because initial prediction is already set

    delta_t = t[k] - t[k - 1]  # time step
    theta = wraptopi(x_check[2, 0])

    # Odometry input
    v_k = v[k - 1]
    om_k = om[k - 1]

    # 1. Update state with odometry readings
    x_check[0] += v_k * np.cos(theta) * delta_t
    x_check[1] += v_k * np.sin(theta) * delta_t
    x_check[2] = wraptopi(x_check[2] + om_k * delta_t)

    # 2. Compute Jacobian of the motion model (F_k)
    F_k = np.array([
        [1, 0, -v_k * np.sin(theta) * delta_t],
        [0, 1, v_k * np.cos(theta) * delta_t],
        [0, 0, 1]
    ])

    # 3. Compute Jacobian with respect to noise (L_k)
    L_k = np.array([
        [np.cos(theta) * delta_t, 0],
        [np.sin(theta) * delta_t, 0],
        [0, delta_t]
    ])

    # 4. Predict covariance
    P_check = F_k @ P_check @ F_k.T + L_k @ Q_km @ L_k.T

    # 5. Correction step if landmark measurement is available
if k < l.shape[1] and k < len(r) and k < len(b) and not np.isnan(b[k]):
        # Pass the correct-sized landmark position and sensor measurements
        x_check, P_check = measurement_update(l[:, k], r[k], b[k], P_check, x_check)

    # 6. Save the predicted state and covariance

# Ensure that x_check is a column vector with 3 elements
if x_check.shape != (3, 1) and x_check.shape != (3,):  # Check both shapes: column vector or flat vector
    raise ValueError(f"x_check has incorrect shape: {x_check.shape}, expected (3, 1) or (3,)")

# Reshape x_check if necessary (convert from column vector to flat vector)
x_check_flat = x_check.flatten()  # This ensures x_check is shape (3,)

# Now assign to the state estimate if the bearing measurement b[k] is valid
# Check if any of the elements in b[k] are NaN
if not np.any(np.isnan(b[k])):
    x_est[k, :] = x_check_flat  # Use the flattened version (shape (3,))
    P_est[k, :, :] = P_check    # Save as row vector

  


## Plot the resulting state estimates:
e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(x_est[:, 0], x_est[:, 1])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Estimated trajectory')
plt.show()

e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(t, x_est[:, 2])
ax.set_xlabel('Time [s]')
ax.set_ylabel('theta [rad]')
ax.set_title('Estimated orientation')
plt.show()
