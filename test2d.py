from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

from bspline import *

def plot(bs,ax,i,j):
    domain = np.linspace(*bs.domain,1000)
    eval = np.vectorize(bs.eval2d)

    pl1 = ax[i,j].plot(bs.X, bs.Y, linestyle='dashed', marker='o', label='control points')
    pl2 = ax[i,j].plot(*eval(bs.knots), linestyle='none', marker='+', label='knots')
    pl3 = ax[i,j].plot(*eval(domain), label='b-spline curve')

def main():
    points = [(0,1), (1,4), (2,5), (3,3), (4,2), (5,4)]
    # points = [(0,0), (1,3), (3,5), (5,4), (6,1)]
    # points = [(0,0), (0,1), (1,1), (1,0), (0.2,0), (0.2,0.8), (0.8,0.8), (0.8,0.2), (0.4,.2), (0.4,0.6), (0.6,0.6), (0.6,0.4), (0.5,0.4), (0.5,0.5)]
    # points = [(0,0), (0,1), (1,1), (1,0)]
    p = 3 # b-spline degree (0 <= p <= n - 1), where n=len(points)

    fig, ax = plt.subplots(nrows=2, ncols=2)

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
