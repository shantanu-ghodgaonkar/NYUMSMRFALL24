import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.optimize import minimize
import sympy as sp
from qpsolvers import solve_qp

lim = 5
REAL_NUMS = np.linspace(-lim, lim, 500)
x = np.meshgrid(REAL_NUMS, REAL_NUMS)


def plot_fx_hx():
    f = (((x[0] - 1)**2) + ((x[1] - 1)**2))
    h1 = x[0] + x[1]
    h2 = x[1] - x[0]
    h3 = x[0] - x[1]
    h4 = -x[1] - x[0]

    intersection = np.logical_and.reduce((h1 < 1, h2 < 1, h3 < 1, h4 < 1))
    # Create a 3D plot
    fig = plt.figure()

    # Filled contour plots for each function
    plt.contourf(x[0], x[1], f, levels=[-lim, lim],
                 colors=['green'], alpha=0.3, label='f')
    plt.contourf(x[0], x[1], h1, levels=[-lim, 1],
                 colors=['blue'], alpha=0.3, label='h1')
    plt.contourf(x[0], x[1], h2, levels=[-lim, 1],
                 colors=['yellow'], alpha=0.3, label='h2')
    plt.contourf(x[0], x[1], h3, levels=[-lim, 1],
                 colors=['red'], alpha=0.3, label='h3')
    plt.contourf(x[0], x[1], h4, levels=[-lim, 1],
                 colors=['purple'], alpha=0.3, label='h4')
    plt.contourf(x[0], x[1], intersection, levels=[0.5, 1],
                 colors=['cyan'], alpha=0.6)

    # Add contour lines for labeling
    cont_f = plt.contour(x[0], x[1], f, levels=[5],
                         colors='green', linestyles='--')
    cont_h1 = plt.contour(x[0], x[1], h1, levels=[1],
                          colors='blue', linestyles='--')
    cont_h2 = plt.contour(x[0], x[1], h2, levels=[1],
                          colors='yellow', linestyles='--')
    cont_h3 = plt.contour(x[0], x[1], h3, levels=[1],
                          colors='red', linestyles='--')
    cont_h4 = plt.contour(x[0], x[1], h4, levels=[1],
                          colors='purple', linestyles='--')

    # Label the contours with the function names directly on the plot
    plt.clabel(cont_f, inline=True, fontsize=10, fmt={
               cont_f.levels[0]: r'(x - 1)^2 + (y - 1)^2'})
    plt.clabel(cont_h1, inline=True, fontsize=10,
               fmt={cont_h1.levels[0]: r'x + y = 1'})
    plt.clabel(cont_h2, inline=True, fontsize=10,
               fmt={cont_h2.levels[0]: r'y - x = 1'})
    plt.clabel(cont_h3, inline=True, fontsize=10,
               fmt={cont_h3.levels[0]: r'x - y = 1'})
    plt.clabel(cont_h4, inline=True, fontsize=10,
               fmt={cont_h4.levels[0]: r'-y - x = 1'})

    # Step 4: Customize the plot
    # plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Overlap of 5 Functions')
    # plt.colorbar()
    plt.grid(True)

    # Step 5: Create custom legend handles
    green_patch = mpatches.Patch(
        color='green', alpha=0.3, label=r'(x - 1)^2 + (y - 1)^2')
    blue_patch = mpatches.Patch(
        color='blue', alpha=0.3, label=r'x + y < 1')
    yellow_patch = mpatches.Patch(
        color='yellow', alpha=0.3, label=r'y - x < 1')
    red_patch = mpatches.Patch(
        color='red', alpha=0.3, label=r'x - y < 1')
    purple_patch = mpatches.Patch(
        color='purple', alpha=0.3, label=r'-y -x <1')
    cyan_patch = mpatches.Patch(
        color='cyan', alpha=0.6, label=r'intersection')

    # Step 6: Add the legend to the plot
    plt.legend(handles=[green_patch, blue_patch,
               yellow_patch, red_patch, purple_patch, cyan_patch])

    # Show the plot
    plt.show()


if __name__ == '__main__':
    # plot_fx_hx()

    # Quadratic term matrix (P)
    P = np.array([[2, 0],
                  [0, 2]])

    # Linear term vector (q) reshaped as a column vector
    q = np.array([-2, -2])

    # Constraint matrix (G) and RHS vector (h)
    G = np.array([[1, 1],
                  [-1, 1],
                  [1, -1],
                  [-1, -1]])
    h = np.array([1, 1, 1, 1])

    # Solve the QP problem using qpsolver
    x = solve_qp(P=P, q=q, G=G, h=h, solver="clarabel")
    print(f"QP solution: {x = }")

    # Get the optimal x and y values
    x_opt, y_opt = x[0], x[1]
    print("Optimal solution for x:", x_opt)
    print("Optimal solution for y:", y_opt)

    # Check the values of the constraints at the optimal solution
    h1 = x_opt + y_opt - 1
    h2 = y_opt - x_opt - 1
    h3 = x_opt - y_opt - 1
    h4 = -x_opt - y_opt - 1

    print(f"h1(x*, y*) = {h1}")
    print(f"h2(x*, y*) = {h2}")
    print(f"h3(x*, y*) = {h3}")
    print(f"h4(x*, y*) = {h4}")

    # Use complementary slackness to determine mu values
    mu_1 = 0 if h1 < 0 else "non-zero"
    mu_2 = 0 if h2 < 0 else "non-zero"
    mu_3 = 0 if h3 < 0 else "non-zero"
    mu_4 = 0 if h4 < 0 else "non-zero"

    print(
        f"Estimated Lagrange multipliers: mu1={mu_1}, mu2={mu_2}, mu3={mu_3}, mu4={mu_4}")

    # Use stationarity to solve for the non-zero mu values
    if mu_1 != 0 or mu_3 != 0 or mu_2 != 0 or mu_4 != 0:
        # Stationarity equations
        eq1 = 2 * (x_opt - 1) + (mu_1 if mu_1 != 0 else 0) + (mu_3 if mu_3 !=
                                                              0 else 0) - (mu_2 if mu_2 != 0 else 0) - (mu_4 if mu_4 != 0 else 0)
        eq2 = 2 * (y_opt - 1) + (mu_1 if mu_1 != 0 else 0) + (mu_2 if mu_2 !=
                                                              0 else 0) - (mu_3 if mu_3 != 0 else 0) - (mu_4 if mu_4 != 0 else 0)

        print(f"Stationarity equation results: eq1={eq1}, eq2={eq2}")
