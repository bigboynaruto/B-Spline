import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from bspline import *

def plot(bs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    domain = np.linspace(*bs.domain,1000)
    eval = np.vectorize(bs.eval)

    plt.plot(bs.X, bs.Y, bs.Z, linestyle='dashed', marker='o', label='points')
    plt.plot(*eval(bs.knots), linestyle='none', marker='+', label='knots')
    plt.plot(*eval(domain), label='b-spline')
    plt.show()

def main():
    points = [(0,0,0), (2,3,1), (2,5,4), (6,2,1), (5,7,3), (4,2,8), (0,3,1)]
    #points = [(0,0,0), (1,3,0), (3,5,0), (5,4,0), (6,1,0)]
    #points = [(0,1), (1,4), (2,5), (3,3), (4,2), (5,4)]
    #points = [(0,0), (0,1), (1,1), (1,0), (0.2,0), (0.2,0.8), (0.8,0.8), (0.8,0.2), (0.4,.2), (0.4,0.6), (0.6,0.6), (0.6,0.4), (0.5,0.4), (0.5,0.5)]
    #points = [(0,0), (0,1), (1,1), (1,0)]
    p = 3# b-spline degree (0 <= p <= n - 1)

    plot(BSpline(points, p))
    plot(OpenBSpline(points, p))
    plot(ClosedBSpline(points, p))

if __name__ == '__main__':
    main()
