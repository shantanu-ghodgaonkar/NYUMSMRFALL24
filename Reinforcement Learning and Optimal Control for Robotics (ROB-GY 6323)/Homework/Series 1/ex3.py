# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
import numpy as np
# from scipy.optimize import minimize
# import sympy as sp
from qpsolvers import solve_qp

# lim = 5
# REAL_NUMS = np.linspace(-lim, lim, 500)
# x = np.meshgrid(REAL_NUMS, REAL_NUMS)


# def plot_fx_gx():
#     f = (((x[0] - 1)**2) + ((x[1] - 1)**2))
#     h1 = x[0] + x[1]
#     h2 = x[1] - x[0]
#     h3 = x[0] - x[1]
#     h4 = -x[1] - x[0]

#     intersection = np.logical_and.reduce((h1 < 1, h2 < 1, h3 < 1, h4 < 1))
#     # Create a 3D plot
#     fig = plt.figure()

#     # Filled contour plots for each function
#     plt.contourf(x[0], x[1], f, levels=[-lim, lim],
#                  colors=['green'], alpha=0.3, label='f')
#     plt.contourf(x[0], x[1], h1, levels=[-lim, 1],
#                  colors=['blue'], alpha=0.3, label='h1')
#     plt.contourf(x[0], x[1], h2, levels=[-lim, 1],
#                  colors=['yellow'], alpha=0.3, label='h2')
#     plt.contourf(x[0], x[1], h3, levels=[-lim, 1],
#                  colors=['red'], alpha=0.3, label='h3')
#     plt.contourf(x[0], x[1], h4, levels=[-lim, 1],
#                  colors=['purple'], alpha=0.3, label='h4')
#     plt.contourf(x[0], x[1], intersection, levels=[0.5, 1],
#                  colors=['cyan'], alpha=0.6)

#     # Add contour lines for labeling
#     cont_f = plt.contour(x[0], x[1], f, levels=[5],
#                          colors='green', linestyles='--')
#     cont_h1 = plt.contour(x[0], x[1], h1, levels=[1],
#                           colors='blue', linestyles='--')
#     cont_h2 = plt.contour(x[0], x[1], h2, levels=[1],
#                           colors='yellow', linestyles='--')
#     cont_h3 = plt.contour(x[0], x[1], h3, levels=[1],
#                           colors='red', linestyles='--')
#     cont_h4 = plt.contour(x[0], x[1], h4, levels=[1],
#                           colors='purple', linestyles='--')

#     # Label the contours with the function names directly on the plot
#     plt.clabel(cont_f, inline=True, fontsize=10, fmt={
#                cont_f.levels[0]: r'(x - 1)^2 + (y - 1)^2'})
#     plt.clabel(cont_h1, inline=True, fontsize=10,
#                fmt={cont_h1.levels[0]: r'x + y = 1'})
#     plt.clabel(cont_h2, inline=True, fontsize=10,
#                fmt={cont_h2.levels[0]: r'y - x = 1'})
#     plt.clabel(cont_h3, inline=True, fontsize=10,
#                fmt={cont_h3.levels[0]: r'x - y = 1'})
#     plt.clabel(cont_h4, inline=True, fontsize=10,
#                fmt={cont_h4.levels[0]: r'-y - x = 1'})

#     # Step 4: Customize the plot
#     # plt.legend()
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title('Overlap of 5 Functions')
#     # plt.colorbar()
#     plt.grid(True)

#     # Step 5: Create custom legend handles
#     green_patch = mpatches.Patch(
#         color='green', alpha=0.3, label=r'(x - 1)^2 + (y - 1)^2')
#     blue_patch = mpatches.Patch(
#         color='blue', alpha=0.3, label=r'x + y < 1')
#     yellow_patch = mpatches.Patch(
#         color='yellow', alpha=0.3, label=r'y - x < 1')
#     red_patch = mpatches.Patch(
#         color='red', alpha=0.3, label=r'x - y < 1')
#     purple_patch = mpatches.Patch(
#         color='purple', alpha=0.3, label=r'-y -x <1')
#     cyan_patch = mpatches.Patch(
#         color='cyan', alpha=0.6, label=r'intersection')

#     # Step 6: Add the legend to the plot
#     plt.legend(handles=[green_patch, blue_patch,
#                yellow_patch, red_patch, purple_patch, cyan_patch])

#     # Show the plot
#     plt.show()


if __name__ == '__main__':
    # plot_fx_hx()

    # Quadratic term matrix (Q)
    Q = np.array([[100, 2, 1], [2, 10, 3], [1, 3, 1]])

    # Constraint matrix (A) and RHS vector (b)
    A = np.ones((1, 3))
    b = np.array([1])

    # Stacking the LHS matrices
    lhs_a = np.vstack((np.hstack((Q, A.T)), np.hstack((A, [[0]]))))

    # stacking the RHS matrices
    rhs_b = np.vstack((np.zeros((3, 1)), [1]))

    # Solve the QP problem using numpy.linalg.solve
    x = np.linalg.solve(lhs_a, rhs_b)
    print(f"QP solution: {x = }")
