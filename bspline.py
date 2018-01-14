from __future__ import division

import numpy as np
from numpy.polynomial.polynomial import polyval2d,polygrid2d
import matplotlib.pyplot as plt
from functools import reduce

class Interval(object):
    """Numeric interval [a,b].

    Attributes:
        a -- min value
        b -- max value
    """
    def __init__(self,a,b):
        super(Interval, self).__init__()

        assert(a <= b)

        self.a = a
        self.b = b

    def __contains__(self,x):
        return self.a <= x <= self.b

    def __getitem__(self,i):
        if i == 0:
            return self.a
        if i == 1:
            return self.b
        raise IndexError

    def __eq__(self,another):
        return self.a == another.a and self.b == another.b

    def __lt__(self,x):
        return x < self.a

    def __gt__(self,x):
        return x > self.b

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return '[%f,%f]' % (self.a,self.b)

class BSpline(object):
    """Clamped b-spline is connected to the first and last points.

    Attributes:
        points -- interpolation points
        p -- spline degree
        C -- evaluation functions
        N -- basis functions
        U -- knots

    TODO: return first or last function if value is not in range
    """
    def __init__(self,points,p):
        super(BSpline, self).__init__()

        assert(2 <= len(points))
        assert(0 <= p and p <= len(points)-1)

        self.points = np.asarray(points)
        self.p = p

        self.U = self._uniform_knots(self.n, self.p)
        self.N = self._basis(self.U, self.p)

        self.C = {}
        for i,poly in self.N.items():
            self.C[i] = [reduce(np.polyadd, (np.polymul(p,c) for p,c in zip(poly,coord))) for coord in np.transpose(self.points)]

    def _uniform_knots(self,n,p):
        U = [(i+1) / (n-p) for i in range(n-p-1)]
        return np.concatenate([np.zeros(p+1), U, np.full(p+1, 1)])

    def _Nij(self,N,i,j,U):
        z = np.zeros(1)
        if all(N[i] == z) and all(N[i+1] == z):
            return z
        elif all(N[i] == z):
            return np.polymul(N[i+1], [-1, U[i+j+1]])/(U[i+j+1]-U[i+1])
        elif all(N[i+1] == z):
            return np.polymul(N[i], [1, -U[i]])/(U[i+j]-U[i])
        return np.polyadd(np.polymul(N[i], [1, -U[i]]) / (U[i+j]-U[i]), np.polymul(N[i+1], [-1, U[i+j+1]])/(U[i+j+1]-U[i+1]))

    def _basis(self,U,p):
        res = {}
        for i in (i for i in range(len(U)-1) if U[i] != U[i+1]):
            N = [0]*(len(U)-1)
            N[i] = 1
            for j in range(1,p+1):
                N[:] = [self._Nij(N,i,j,U) for i in range(len(N)-1)]
            res[Interval(U[i], U[i+1])] = np.asarray(N)
        return res

    def _find_interval(self,u):
        for i,c in self.C.items():
            if u in i:
                return c
        raise Exception

    def eval2d(self,u):
        f = self._find_interval(u)
        return tuple(np.polyval(c,u) for c in f[:2])

    def eval(self,u):
        f = self._find_interval(u)
        return tuple(np.polyval(c,u) for c in f)

    def evalx(self,u):
        f = self._find_interval(u)
        return np.polyval(f[0],u)

    def evaly(self,u):
        f = self._find_interval(u)
        return np.polyval(f[1],u)

    def evalz(self,u):
        #if (len(self.points[0]) < 3):
        #    return 0
        f = self._find_interval(u)
        return np.polyval(f[2],u)

    """x-values of control points."""
    @property
    def X(self):
        return [p[0] for p in self.points]

    """y-values of control points."""
    @property
    def Y(self):
        return [p[1] for p in self.points]

    """z-values of control points."""
    @property
    def Z(self):
        if (len(self.points[0]) > 2):
            return [p[2] for p in self.points]
        return np.zeros(len(self.points))

    """B-Spline parameter domain."""
    @property
    def domain(self):
        return Interval(np.min(self.U),np.max(self.U))

    """Number of points."""
    @property
    def n(self):
        return len(self.points)

    """Number of knots."""
    @property
    def m(self):
        return self.n + self.p + 1

    """Knots."""
    @property
    def knots(self):
        return np.unique(self.U)


class OpenBSpline(BSpline):
    """Open b-spline doesn't touch first and last points."""
    def __init__(self,points,p):
        super(OpenBSpline, self).__init__(points,p)

    def _uniform_knots(self,n,p):
        return np.asarray([i / (n+p) for i in range(n+p+1)])

    def _basis(self,U,p):
        res = {}
        for i in (i for i in range(len(U)-1) if p <= i < len(U) - p and U[i] != U[i+1]):
            N = [0]*(len(U)-1)
            N[i] = 1
            for j in range(1,p+1):
                N[:] = [self._Nij(N,i,j,U) for i in range(len(N)-1)]
            res[Interval(U[i], U[i+1])] = N
        return res

    @property
    def domain(self):
        return Interval(self.U[self.p],self.U[self.m-self.p-1])

    @property
    def knots(self):
        return np.unique(self.U[self.p:self.n+1])

class ClosedBSpline(OpenBSpline):
    """Closed b-spline with joined start and end."""
    def __init__(self,points,p):
        points.extend(points[:p])
        super(ClosedBSpline, self).__init__(points,p)

class BSplineSurface(BSpline):
    """Clamped b-spline surface."""
    def __init__(self,points,p,q):
        assert(0 <= p <= len(points) - 1)
        assert(0 <= q <= len(points[0]) - 1)

        self.points = np.asarray(points)
        self.p = p
        self.q = q

        self.U = self._uniform_knots(self.n, p)
        self.V = self._uniform_knots(self.l, q)

        self.N = self._basis(self.U, p)
        self.M = self._basis(self.V, q)

        self.C = {}
        for i1,N in self.N.items():
            for i2,M in self.M.items():
                C = np.zeros((self.n,self.l,3))
                for pu in range(len(N)):
                    for pv in range(len(M)):
                        n,l = len(N[pu]),len(M[pv])
                        for i in range(n):
                            for j in range(l):
                                C[n-i-1,l-j-1] += N[pu][i] * M[pv][j] * self.points[pu][pv]
                self.C[i1,i2] = C

    def _find_interval(self,u,v):
        for i,c in self.C.items():
            if u in i[0] and v in i[1]:
                return c
        raise Exception

    def eval(self,u,v):
        c = self._find_interval(u,v)
        return tuple(polyval2d(u,v,c))

    def eval2d(self,u,v):
        c = self._find_interval(u, v)
        return tuple(polyval2d(u,v,np.asarray([[[i[0],i[1]] for i in j] for j in c])))

    def evalx(self,u,v):
        c = self._find_interval(u, v)
        return polyval2d(u,v,[[i[0] for i in j] for j in c])

    def evaly(self,u,v):
        c = self._find_interval(u, v)
        return polyval2d(u,v,[[i[1] for i in j] for j in c])

    def evalz(self,u,v):
        c = self._find_interval(u, v)
        return polyval2d(u,v,[[i[2] for i in j] for j in c])

    """x-values of control points."""
    @property
    def X(self):
        return [[x[0] for x in xx] for xx in self.points]

    """y-values of control points."""
    @property
    def Y(self):
        return [[x[1] for x in xx] for xx in self.points]

    """z-values of control points."""
    @property
    def Z(self):
        return [[x[2] for x in xx] for xx in self.points]

    """B-Spline parameter domain."""
    @property
    def domain(self):
        return Interval(np.min(self.U),np.max(self.U)),Interval(np.min(self.V),np.max(self.V))

    """Number of points."""
    @property
    def l(self):
        return len(self.points[0])

    """Number of knots."""
    @property
    def k(self):
        return self.l + self.q + 1

    """Knots."""
    @property
    def knots(self):
        return self.U,self.V

class OpenBSplineSurface(OpenBSpline,BSplineSurface):
    """Open b-spline surface doesn't touch first and last lines in grid."""
    def __init__(self,points,p,q):
        super(OpenBSpline, self).__init__(points,p,q)

    @property
    def domain(self):
        return Interval(self.U[self.p],self.U[self.n]),Interval(self.V[self.q],self.V[self.l])

    @property
    def knots(self):
        return np.unique(self.U[self.p:self.n+1]),np.unique(self.V[self.q:self.l+1])

class ClosedBSplineSurface(OpenBSplineSurface):
    """Closed b-spline surface with joined start and end."""
    def __init__(self,points,p,q):
        points =[ps + ps[:p] for ps in points]
        points.extend(points[:p])
        super(ClosedBSplineSurface, self).__init__(points,p,q)
