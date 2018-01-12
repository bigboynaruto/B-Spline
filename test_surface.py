import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from bspline import *

def plot(bss):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    d1,d2 = bss.domain
    u,v = np.linspace(d1[0],d1[1],50),np.linspace(d2[0],d2[1],50)
    eval = np.vectorize(bss.eval)

    u,v = np.meshgrid(u,v)
    X,Y,Z = eval(u,v)

    U,V = np.meshgrid(*bss.knots)
    _X,_Y,_Z = eval(U,V)

    #ax.plot(np.ravel(_X), np.ravel(_Y), np.ravel(_Z), linestyle='none', marker='x', label='knots')
    ax.plot_surface(X, Y, Z, color='orange', label='B-spline surface')
    ax.plot_wireframe(np.array(bss.X), np.array(bss.Y), np.array(bss.Z), label='control points')

    # ax.legend() # error

    return ax

def torus(R=5,r=1,n=10,m=10):
    phi,theta = np.linspace(0,np.pi,n),np.linspace(0,2*np.pi,m)
    phi[-1],theta[-1] = np.pi,2*np.pi
    phi,theta = np.meshgrid(phi,theta)

    X = (R + r*np.cos(phi)) * np.cos(theta)
    Y = (R + r*np.cos(phi)) * np.sin(theta)
    Z = r * np.sin(phi)

    return X,Y,Z

def sphere(r=1,n=10,m=10):
    phi,theta = np.linspace(0,np.pi,n),np.linspace(0,2*np.pi,m)
    phi[-1],theta[-1] = np.pi,2*np.pi
    phi,theta = np.meshgrid(phi,theta)

    X = r * np.sin(phi) * np.cos(theta)
    Y = r * np.sin(phi) * np.sin(theta)
    Z = r * np.cos(phi)

    return X,Y,Z

def something(a=-6,b=6):
    assert(a < b)

    X = range(a,b)
    Y = range(a,b)
    X,Y = np.meshgrid(X,Y)

    Z = np.sin(np.sqrt(X*X + Y*Y))

    return X,Y,Z

def mobius():
    theta = np.linspace(0, 2 * np.pi, 10)
    w = np.linspace(-0.25, 0.25, 8)
    w, theta = np.meshgrid(w, theta)

    phi = 0.5 * theta

    r = 1 + w * np.cos(phi)

    X = r * np.cos(theta)
    Y = r * np.sin(theta)
    Z = w * np.sin(phi)

    return X,Y,Z

def main():
    p,q = 3,3 # b-spline degree (0 <= p,q <= n - 1)

    #X,Y,Z = torus()
    #X,Y,Z = sphere()
    #X,Y,Z = something()
    X,Y,Z = mobius()
    points = [[[x,y,z] for x,y,z in zip(xx,yy,zz)] for xx,yy,zz in zip(X,Y,Z)]

    ax = plot(BSplineSurface(points,p,q)); ax.set_title('Clamped B-spline surface')
    #ax = plot(OpenBSplineSurface(points,p,q)); ax.set_title('Open B-spline surface')
    #ax = plot(ClosedBSplineSurface(points,p,q)); ax.set_title('Closed B-spline surface')

    plt.show()

if __name__ == '__main__':
    main()
