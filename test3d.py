import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from bspline import *

def plot(bs,ax,i,j):
    domain = bs.domain.linspace(1000)
    eval = np.vectorize(bs.eval)

    ax[i,j].plot(bs.X, bs.Y, bs.Z, linestyle='dashed', marker='o', label='control points')
    ax[i,j].plot(*eval(bs.knots), linestyle='none', marker='+', label='knots')
    ax[i,j].plot(*eval(domain), label='b-spline curve')

def main():
    points = [(0,0,0), (2,3,1), (2,5,4), (6,2,1), (5,7,3), (4,2,8), (0,3,1)]
    p = 3 # b-spline degree (0 <= p <= n - 1), where n=len(points)

    fig, ax = plt.subplots(nrows=2, ncols=2, subplot_kw={'projection': '3d'})

    plot(BSpline(points, p), ax, 0, 0)
    ax[0,0].set_title('Clamped B-spline')

    plot(OpenBSpline(points, p), ax, 0, 1)
    ax[0,1].set_title('Open B-spline')

    plot(ClosedBSpline(points, p), ax, 1, 0)
    ax[1,0].set_title('Closed B-spline')

    handles, labels = ax[0,0].get_legend_handles_labels()
    ax[1,1].legend(handles, labels, loc='center')
    ax[1,1].set_axis_off()

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
