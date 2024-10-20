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
""" print(v.shape)
print(len(v))
print(d.shape)
print(d[0]) """

## Initializing Parameters
v_var = 0.01  # translation velocity variance
om_var = 0.01  # rotational velocity variance
r_var = 0.01   # range measurements variance (tuned)
b_var = 10    # bearing measurement variance (tuned)

Q_km = np.diag([v_var, om_var]) # input noise covariance
cov_y = np.diag([r_var, b_var])  # measurement noise covariance

x_est = np.zeros([len(v), 3])  # estimated states, x, y, and theta
P_est = np.zeros([len(v), 3, 3])  # state covariance matrices

x_est[0] = np.array([x_init, y_init, th_init]) # initial state 
P_est[0] = np.diag([1, 1, 0.01]) # initial state covariance

"""
print(Q_km.shape)
print(Q_km)
print(cov_y.shape)
print(cov_y)
print(P_est.shape)
print(P_est[0])  """


# Wraps angle to (-pi, pi] range
def wraptopi(angle):

    return (angle + np.pi) % (2 * np.pi) - np.pi




## Correction Step
def measurement_update(lk, rk, bk, P_check, x_check, d, cov_y):
    # Extract current robot state (x, y, theta)
    x_k = x_check[0]
    y_k = x_check[1]
    theta_k = wraptopi(x_check[2])

    # Landmark position
    x_l = lk[0]
    y_l = lk[1]

    # Expected range and bearing
    d_x = x_l - (x_k + d * np.cos(theta_k))
    d_y = y_l - (y_k + d * np.sin(theta_k))
    r_hat = np.sqrt(d_x**2 + d_y**2)
    phi_hat = np.arctan2(d_y, d_x) - theta_k
    phi_hat = wraptopi(phi_hat)

    z_hat = np.array([r_hat, phi_hat])
    z_k = np.array([rk, wraptopi(bk)])

    r = r_hat
    H_k = np.zeros((2, 3))
    H_k[0, 0] = -d_x / r
    H_k[0, 1] = -d_y / r
    H_k[0, 2] = d * (d_x * np.sin(theta_k) - d_y * np.cos(theta_k)) / r
    H_k[1, 0] = d_y / r**2
    H_k[1, 1] = -d_x / r**2
    H_k[1, 2] = -1 - d * (d_y * np.sin(theta_k) + d_x * np.cos(theta_k)) / r**2

    S_k = H_k.dot(P_check).dot(H_k.T) + cov_y
    K_k = P_check.dot(H_k.T).dot(np.linalg.inv(S_k))

    y_k = z_k - z_hat
    y_k[1] = wraptopi(y_k[1])

    x_check = x_check + K_k.dot(y_k)
    x_check[2] = wraptopi(x_check[2])

    P_check = (np.identity(3) - K_k.dot(H_k)).dot(P_check)

    return x_check, P_check



## Prediction Step
P_check = P_est[0]
x_check = x_est[0, :].reshape(3, 1)

for k in range(1, len(t)):  # start at 1 because initial prediction is already set

    delta_t = t[k] - t[k - 1]  # time step
     
    theta = wraptopi(x_check[2, 0])

    # Odometry input
    v_k = v[k-1] 
    om_k = om[k-1]

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
# Ensure x_check is a 1D array of shape (3,)
    if x_check.shape == (3, 1):
        x_check_flat = x_check.flatten()  # Flatten (3,1) into (3,)
    elif x_check.shape == (3,):
        x_check_flat = x_check  # Already in correct shape
    else:
        raise ValueError(f"x_check has incorrect shape: {x_check.shape}, expected (3,1) or (3,)")

# Save the updated state
    x_est[k, :] = x_check_flat  # Store the flattened (3,)

# Ensure P_check is a 3x3 matrix
    if P_check.shape == (3, 3):
        P_est[k, :, :] = P_check  # Store the covariance matrix
    else:
        raise ValueError(f"P_check has incorrect shape: {P_check.shape}, expected (3,3)")


    # 6. Save the predicted state and covariance

    x_est[k, :] = x_check.flatten()  # Save the updated state (flatten to ensure it's 1D)
    P_est[k, :, :] = P_check         # Save the updated covariance

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
