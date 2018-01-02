from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from sympy import *
from sympy.utilities.lambdify import lambdify

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

    def __eq__(self,another):
        return self.a == another.a and self.b == another.b

    def __lt__(self,x):
        return x < self.a

    def __gt__(self,x):
        return x > self.a

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

        self.points = points
        self.p = p
        self._gen_U()
        self._gen_N()
        self._gen_C()

    def _gen_U(self):
        U = [(i+1) / (self.n-self.p) for i in range(self.n-self.p-1)]
        self.U = np.concatenate([np.zeros(self.p+1), U, np.full(self.p+1, 1)])

    def _Nij(self,u,N,i,j):
        U = self.U
        if N[i] == 0 and N[i+1] == 0:
            return 0
        elif N[i] == 0:
            return N[i+1]*(U[i+j+1]-u)/(U[i+j+1]-U[i+1])
        elif N[i+1] == 0:
            return N[i]*(u-U[i])/(U[i+j]-U[i])
        return N[i]*(u-U[i])/(U[i+j]-U[i]) + N[i+1]*(U[i+j+1]-u)/(U[i+j+1]-U[i+1])

    def _gen_N(self):
        u = Symbol('u', rational=True)
        self.N = {}
        for i in (i for i in range(len(self.U)-1) if self.U[i] != self.U[i+1]):
            currN = np.zeros(len(self.U)-1)
            currN[i] = 1
            for j in range(1,self.p+1):
                currN = [self._Nij(u,currN,i,j) for i in range(len(currN)-1)]
            self.N[Interval(self.U[i], self.U[i+1])] = currN

    def _gen_C(self):
        self.C = {}
        u = Symbol('u', rational=True)
        for i,v in self.N.items():
            self.C[i] = [lambdify(u,f,modules='numpy') for f in np.dot(v,self.points)]

    def eval2d(self,u):
        return self.evalx(u),self.evaly(u)

    def eval(self,u):
        return self.evalx(u),self.evaly(u),self.evalz(u)

    def evalx(self,u):
        for i,c in self.C.items():
            if u in i:
                return c[0](u)
        return 0

    def evaly(self,u):
        for i,c in self.C.items():
            if u in i:
                return c[1](u)
        return 0

    def evalz(self,u):
        if (len(self.points[0]) < 3):
            return 0
        for i,c in self.C.items():
            if u in i:
                return c[2](u)
        return 0

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
        return np.min(self.U),np.max(self.U)

    """Number of points."""
    @property
    def n(self):
        return len(self.points)

    """Number of knots."""
    @property
    def m(self):
        return self.n + self.p + 1

    """Knots used in evaluation."""
    @property
    def knots(self):
        return self.U


class OpenBSpline(BSpline):
    """Open b-spline doesn't touch first and last points."""
    def __init__(self,points,p):
        super(OpenBSpline, self).__init__(points,p)

    def _gen_U(self):
        self.U = np.array([i / (self.m-1) for i in range(self.m)])

    def _gen_N(self):
        u = Symbol('u', rational=True)
        self.N = {}
        for i in (i for i in range(len(self.U)-1) if self.U[i] != self.U[i+1]):
            if i < self.p or i >= len(self.U) - self.p:
                continue
            currN = np.zeros(len(self.U)-1)
            currN[i] = 1
            for j in range(1,self.p+1):
                currN = [self._Nij(u,currN,i,j) for i in range(len(currN)-1)]
            self.N[Interval(self.U[i], self.U[i+1])] = currN

    @property
    def domain(self):
        return self.U[self.p],self.U[self.m-self.p-1]

    @property
    def knots(self):
        return np.unique(self.U[self.p:self.m-self.p])

class ClosedBSpline(OpenBSpline):
    """Closed b-spline with joined start and end."""
    def __init__(self,points,p):
        points.extend(points[:p])
        super(ClosedBSpline, self).__init__(points,p)
