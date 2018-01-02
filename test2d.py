from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

from bspline import *

def plot(bs):
    domain = np.linspace(*bs.domain,1000)
    eval = np.vectorize(bs.eval2d)

    plt.plot(bs.X, bs.Y, linestyle='dashed', marker='o')
    plt.plot(*eval(bs.knots), linestyle='none', marker='+')
    plt.plot(*eval(domain))
    plt.show()

def main():
    points = [(0,0), (1,3), (3,5), (5,4), (6,1)]
    #points = [(0,1), (1,4), (2,5), (3,3), (4,2), (5,4)]
    #points = [(0,0), (0,1), (1,1), (1,0), (0.2,0), (0.2,0.8), (0.8,0.8), (0.8,0.2), (0.4,.2), (0.4,0.6), (0.6,0.6), (0.6,0.4), (0.5,0.4), (0.5,0.5)]
    #points = [(0,0), (0,1), (1,1), (1,0)]
    p = 3# b-spline degree (0 <= p <= n - 1)

    plot(BSpline(points, p))
    plot(OpenBSpline(points, p))
    plot(ClosedBSpline(points, p))

if __name__ == '__main__':
    main()
