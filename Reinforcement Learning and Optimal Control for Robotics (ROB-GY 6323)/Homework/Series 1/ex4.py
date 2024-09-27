import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import quadrotor

# the matrices A and B are already defined in the quadrotor module
# print(f'A =\n {quadrotor.A}')
# print(f'B =\n {quadrotor.B}')

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
plot_folder = "/home/shantanu/Documents/Academics/NYUMSMRFALL24/Reinforcement Learning and Optimal Control for Robotics (ROB-GY 6323)/Homework/Series 1/plots/ex4/"
state_plot_filename = f"{plot_folder}state_plot_{timestamp}.png"
control_ip_filename = f"{plot_folder}octrl_plot_{timestamp}.png"
# write down your code here
N = 500

Q = np.diag([31.25, 3.125, 31.25, 3.125, 0.3125, 0.3125])
R = np.diag([0.75, 0.75])


P = np.zeros((N, 6, 6))
K = np.zeros((N, 2, 6))

u = np.zeros((2, N))
x = np.zeros((6, N))
x_desired = np.array([3., 0., 3., 0., 0., 0.])

P[N-1] = Q

for i in range(N-2, 0, -1):
    lhs_K = R + quadrotor.B.T @ P[i+1] @ quadrotor.B
    rhs_K = quadrotor.B.T @ P[i+1] @ quadrotor.A
    K[i+1] = np.linalg.solve(lhs_K, rhs_K)
    P[i] = Q + quadrotor.A.T @ P[i+1] @ quadrotor.A - \
        quadrotor.A.T @ P[i+1] @ quadrotor.B @ K[i+1]

for i in range(0, N, 1):
    u[:, i] = -K[i] @ (x[:, i] - x_desired)
    if (i+1) != 500:
        x[:, (i+1)] = quadrotor.A @ x[:, i] + quadrotor.B @ u[:, i]

print(f"\n\nFinal State = {np.round(x[:, N-1], 5)}")
print(f"U1 : Max = {round(u[0].max(), 5)}, Min = {round(u[0].min(), 5)}")
print(f"U2 : Max = {round(u[1].max(), 5)}, Min = {round(u[1].min(), 5)}")

# Plot states over time
time = np.arange(N)
# Plot x position
plt.figure(figsize=(20, 10))
plt.title(f"State Plots Q={Q.diagonal()}, R={R.diagonal()}")
plt.subplot(3, 2, 1)
plt.plot(time, x[0, :], label='x position')
plt.xlabel('Time')
plt.ylabel('x')
plt.grid(True)
# Plot x velocity
plt.subplot(3, 2, 2)
plt.plot(time, x[1, :], label='x velocity')
plt.xlabel('Time')
plt.ylabel('x velocity')
plt.grid(True)
# Plot y position
plt.subplot(3, 2, 3)
plt.plot(time, x[2, :], label='y position')
plt.xlabel('Time')
plt.ylabel('y')
plt.grid(True)
# Plot y velocity
plt.subplot(3, 2, 4)
plt.plot(time, x[3, :], label='y velocity')
plt.xlabel('Time')
plt.ylabel('y velocity')
plt.grid(True)
# Plot theta
plt.subplot(3, 2, 5)
plt.plot(time, x[4, :], label='theta')
plt.xlabel('Time')
plt.ylabel('theta')
plt.grid(True)
# Plot theta velocity
plt.subplot(3, 2, 6)
plt.plot(time, x[5, :], label='angular velocity')
plt.xlabel('Time')
plt.ylabel('angular velocity')
plt.grid(True)
plt.tight_layout()
# plt.savefig(state_plot_filename)
plt.show()

# Plot control inputs
plt.figure(figsize=(20, 10))
plt.title(f"Control Inputs Plot Q={Q.diagonal()}, R={R.diagonal()}")
plt.plot(np.arange(N), u[0, :], label='u1')
plt.plot(np.arange(N), u[1, :], label='u2')
plt.xlabel('Time')
plt.ylabel('Control inputs')
plt.legend()
plt.grid(True)
# plt.savefig(control_ip_filename)
plt.show()
