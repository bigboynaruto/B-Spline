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

    def linspace(self,n):
        return np.linspace(self.a, self.b, n)

    def __contains__(self,x):
        return self.a <= x <= self.b

    def __eq__(self,another):
        return self.a == another.a and self.b == another.b

    def __lt__(self,x):
        return x < self.a

    def __gt__(self,x):
        return x > self.b

    def __hash__(self):
        return hash((self.a,self.b))

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
        if not np.any(N[i]) and not np.any(N[i+1]):
            return [0]
        elif not np.any(N[i]):
            return np.polymul(N[i+1], [-1, U[i+j+1]])/(U[i+j+1]-U[i+1])
        elif not np.any(N[i+1]):
            return np.polymul(N[i], [1, -U[i]])/(U[i+j]-U[i])
        return np.polyadd(np.polymul(N[i], [1, -U[i]]) / (U[i+j]-U[i]), np.polymul(N[i+1], [-1, U[i+j+1]])/(U[i+j+1]-U[i+1]))

    def _basis(self,U,p):
        res = {}
        for i in (i for i in range(len(U)-1) if U[i] != U[i+1]):
            N = [[0] for i in range(len(U)-1)]
            N[i][0] = 1
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
        U = self.knots
        return Interval(np.min(U),np.max(U))

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
            N = [[0] for i in range(len(U)-1)]
            N[i][0] = 1
            for j in range(1,p+1):
                N[:] = [self._Nij(N,i,j,U) for i in range(len(N)-1)]
            res[Interval(U[i], U[i+1])] = np.asarray(N)
        return res

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
            C = {}
            for i2,M in self.M.items():
                _C = np.zeros((self.n,self.l,3))

                for us,ps in zip(N,self.points):
                    for vs,p in zip(M,ps):
                        n,l = len(us),len(vs)

                        for i,u in zip(range(n-1,-1,-1),us):
                            for j,v in zip(range(l-1,-1,-1),vs):
                                _C[i,j] += u * v * p
                C[i2] = _C
            self.C[i1] = C

    def _find_interval(self,u,v):
        for i1,C in self.C.items():
            if u in i1:
                for i2,c in C.items():
                    if v in i2:
                        return c
        raise Exception

    def eval(self,u,v):
        c = self._find_interval(u,v)
        return tuple(polyval2d(u,v,c))

    # useless function
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
        U,V = self.knots
        return Interval(np.min(U),np.max(U)),Interval(np.min(V),np.max(V))

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
        return np.unique(self.U),np.unique(self.V)

class OpenBSplineSurface(BSplineSurface,OpenBSpline):
    """Open b-spline surface doesn't touch first and last lines in grid."""
    def __init__(self,points,p,q):
        super(OpenBSplineSurface, self).__init__(points,p,q)

    @property
    def knots(self):
        return np.unique(self.U[self.p:self.n+1]),np.unique(self.V[self.q:self.l+1])

class ClosedBSplineSurface(OpenBSplineSurface):
    """Closed b-spline surface with joined start and end."""
    def __init__(self,points,p,q):
        points =[ps + ps[:q] for ps in points]
        points.extend(points[:p])
        super(ClosedBSplineSurface, self).__init__(points,p,q)
