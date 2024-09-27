import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import sympy as sp
from qpsolvers import solve_qp


lim = 100
REAL_NUMS = np.linspace(-lim, lim, 10)
x = np.meshgrid(REAL_NUMS, REAL_NUMS)


def f3(x):
    return (20*x[0]) + (2*(x[0]**2)) + (4*x[1]) - (2 * (x[1]**2))


if __name__ == "__main__":
    # P = np.array(([[2, 0], [0, -2]]))
    # q = np.array(([[10, 2]])).T
    # res = solve_qp(P=P, q=q, solver='clarabel')
    # print(res)
    # # Create a 3D plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # y = f3(x)
    # # Plot the surface
    # ax.plot_surface(x[0], x[1], y, cmap='viridis')

    # # Add labels and title
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis (f(x, y))')
    # ax.set_title(r'Plot of f(x, y) = $20x + 2x^2 + 4y - 2y^2$')
    # # Save the plot to a file
    # plt.savefig('plots/ex1/ex1cPlot.png')
    # # Show the plot
    # plt.show()

    lim = 10
    REAL_NUMS = np.linspace(-lim, lim, 25)

    P = np.array(([[1, 1, 0], [1, 1, 0], [0, 0, 4]]))
    q = np.array(([[0, 0, -1]])).T
    res = solve_qp(P=P, q=q, solver='clarabel')

    def f6(x): return 0.5 * (x[0] * (1 * x[0] + 1 * x[1]) + x[1] * (
        1 * x[0] + 1 * x[1]) + x[2] * (4 * x[2])) - (0 * x[0] + 0 * x[1] + 1 * x[2])

    print(
        f"\n\nUsing QP Solvers : Minimum value of the function f6 was found to be {f6(res)} at x = {res}")
    res = minimize(f6, x0=[0, 0, 0])
    print(
        f"\n\nUsing scipy.optimize.minimize : Minimum value of the function f6 was found to be {res.fun} at x = {res.x}\n\n")

    x = np.meshgrid(REAL_NUMS, REAL_NUMS, REAL_NUMS)
    y = f6(x)
    # Create a contour plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create filled contours
    img = ax.scatter(x[0], x[1], x[2], c=y, cmap='YlOrRd', alpha=1)

    # # Add a color bar to indicate the values of the contours
    # plt.colorbar()

    # # Add labels, grid, legend, and title
    ax.set_title(r'Plot of f6(x)')  # Title with LaTeX
    ax.set_xlabel(r'$x$')               # X-axis label with LaTeX
    ax.set_ylabel(r'$y$')          # Y-axis label with LaTeX
    ax.set_zlabel(r'$z$')          # Z-axis label with LaTeX
    plt.grid(True)
    # Show the plot
    plt.show()
