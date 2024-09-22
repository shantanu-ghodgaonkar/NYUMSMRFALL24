import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import sympy as sp

lim = 5
REAL_NUMS = np.linspace(0, lim, 500)
x = sp.symbols('x:2')
f1 = -sp.exp(-((x[0] - 1)**2))
f2 = ((1 - x[0])**2) + (100 * ((x[1] - (x[0]**2))**2))
f2_num = sp.lambdify((x, ), f2,  modules='numpy')
f3 = (20 * x[0]) + (2 * (x[0]**2)) + (4 * x[1]) - (2 * (x[1]**2))


# def solve_f1():
#     min_conditions = 0
#     f1_num = sp.lambdify(x[0], f1)
#     res = minimize(f1_num, x0=0)
#     y = [f1.subs({x: i}) for i in REAL_NUMS]
#     nabla_f1 = sp.diff(f1, x)
#     # nabla_y = [nabla_f1.subs({x: i}) for i in REAL_NUMS]
#     hessian_f1 = sp.diff(nabla_f1, x)
#     # hessian_y = [hessian_f1.subs({x: i}) for i in REAL_NUMS]
#     fonc = round(nabla_f1.subs({x: res.x[0]}), 5)
#     # sonc = np.linalg.cholesky(hessian_f1.subs({x: res.x[0]}))
#     sonc = round(hessian_f1.subs({x: res.x[0]}), 5)
#     print(
#         f"1st Order Necessary condition: f'(x_star) = {fonc} => {fonc == 0.0}")
#     print(
#         f"2nd Order Necessary Condition: f''(x_star) = {sonc} => {sonc >= 0.0}")
#     print(
#         f"2nd Order Sufficient Condition: f''(x_star) = {sonc} => {sonc > 0.0}")

#     # Plot with customization
#     plt.plot(REAL_NUMS, y, color='green',
#              linestyle='--', linewidth=2)

#     # Add labels, grid, legend, and title
#     plt.title(r'Plot of $-e^{-(x-1)^2}$')  # Title with LaTeX
#     plt.xlabel(r'$x$')               # X-axis label with LaTeX
#     plt.ylabel(r'$y$')          # Y-axis label with LaTeX
#     plt.grid(True)
#     # plt.legend()
#     # Save the plot to a file
#     plt.savefig('plots/ex1/ex1aPlot.png')
#     # Show the plot
#     plt.show()


def solve_f2():

    res = minimize(f2_num, x0=np.array([0, 0]))
    y = f2_num(np.meshgrid(REAL_NUMS, REAL_NUMS))

    nabla_f1 = sp.diff(f1, x[0])
    # nabla_y = [nabla_f1.subs({x: i}) for i in REAL_NUMS]
    hessian_f1 = sp.diff(nabla_f1, x[0])
    # hessian_y = [hessian_f1.subs({x: i}) for i in REAL_NUMS]
    fonc = round(nabla_f1.subs({x[0]: res.x[0]}), 5)
    # sonc = np.linalg.cholesky(hessian_f1.subs({x: res.x[0]}))
    sonc = round(hessian_f1.subs({x[0]: res.x[0]}), 5)
    print(
        f"1st Order Necessary condition: f'(x_star) = {fonc} => {fonc == 0.0}")
    print(
        f"2nd Order Necessary Condition: f''(x_star) = {sonc} => {sonc >= 0.0}")
    print(
        f"2nd Order Sufficient Condition: f''(x_star) = {sonc} => {sonc > 0.0}")

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(x[0], x[1], y, cmap='viridis')

    # Add labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis (f(x, y))')
    ax.set_title(r'Plot of f(x, y) = $(1 - x)^2 + 100(y - x^2)^2$')
    # Save the plot to a file
    plt.savefig('plots/ex1/ex1bPlot.png')
    # Show the plot
    plt.show()


# def func2(x): return ((1-x[0])**2) + (100 * ((x[1] - (x[0]**2))**2))


# def plot_func2():
#     res = minimize(func2, x0=np.array([0, 0]), bounds=[(0, 0), [-lim, lim]])
#     print(
#         f"Minimum value of x, y = {res.x[0]}, {res.x[1]}\nMinimum value of f(x) = {res.fun}")
#     x, y = np.meshgrid(REAL_NUMS, REAL_NUMS)
#     z = func2([x, y])
#     # Create a 3D plot
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot the surface
#     ax.plot_surface(x, y, z, cmap='viridis')

#     # Add labels and title
#     ax.set_xlabel('X axis')
#     ax.set_ylabel('Y axis')
#     ax.set_zlabel('Z axis (f(x, y))')
#     ax.set_title(r'Plot of f(x, y) = $(1 - x)^2 + 100(y - x^2)^2$')
#     # Save the plot to a file
#     plt.savefig('plots/ex1/ex1bPlot.png')
#     # Show the plot
#     plt.show()


# def plot_func2b():
#     res = minimize(func2, x0=np.array([0, 0]), bounds=[(0, 0), [-lim, lim]])
#     print(
#         f"Minimum value of x, y = {res.x[0]}, {res.x[1]}\nMinimum value of f(x) = {res.fun}")
#     # x, y = np.meshgrid(REAL_NUMS, REAL_NUMS)
#     # z = func2b([x, 0])
#     res = minimize(func1, x0=0)
#     print(f"Minimum value of x = {res.x}\nMinimum value of f(x) = {res.fun}")
#     y = [func2([0, x]) for x in REAL_NUMS]
#     # Plot with customization
#     plt.plot(REAL_NUMS, y, color='green',
#              linestyle='--', linewidth=2)

#     # Add labels, grid, legend, and title
#     plt.title(r'Plot of $-e^{-(x-1)^2}$')  # Title with LaTeX
#     plt.xlabel(r'$x$')               # X-axis label with LaTeX
#     plt.ylabel(r'$y$')          # Y-axis label with LaTeX
#     plt.grid(True)
#     plt.legend()
#     # Save the plot to a file
#     plt.savefig('plots/ex1/ex1aPlot.png')
#     # Show the plot
#     plt.show()
if __name__ == '__main__':
    solve_f2()
    print("DEBUG POINT")
