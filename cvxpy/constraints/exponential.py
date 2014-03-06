"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

from nonlinear import NonlinearConstraint
import math
import cvxopt
import numpy as np

class ExpCone(NonlinearConstraint):
    """An exponential cone constraint.

    K = {(x,y,z) | y > 0, ye^(x/y) <= z} U {(x,y,z) | x <= 0, y = 0, z >= 0}

    Attributes:
        x: The scalar variable x in the exponential cone.
        y: The scalar variable y in the exponential cone.
        z: The scalar variable z in the exponential cone.
        m: The dimension of the constraint.
        _constraints: A list of equality constraints connecting
                      new variables to the given x,y,z.
    """

    # The dimensions of the exponential cone.
    size = (1, 1)

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        super(ExpCone, self).__init__(self._solver_hook,
                                      [self.x, self.y, self.z])

    def __str__(self):
        return "ExpCone(%s, %s, %s)" % (self.x, self.y, self.z)

    @staticmethod
    def _solver_hook(vars_=None, scaling=None):
        """A function used by CVXOPT's nonlinear solver.

        Based on f(x,y,z) = ye^(x/y) - z.

        Args:
            vars_: A cvxopt dense matrix with values for (x,y,z).
            scaling: A scaling for the Hessian.

        Returns:
            _solver_hook() returns the constraint size and a feasible point.
            _solver_hook(x) returns the function value and gradient at x.
            _solver_hook(x, z) returns the function value, gradient,
            and (z scaled) Hessian at x.
        """
        if vars_ is None:
            return ExpCone.size[0], cvxopt.matrix([0.0, 1.0, 1.0])
        # Unpack vars_
        x, y, z = vars_
        # Check out of domain
        ood = (y < 0.0 or y == 0.0 and (x > 0.0 or z < 0.0))
        # Evaluate the function.
        f = y*math.exp(x/y) - z
        print 'x:', x
        print 'y:', y
        print 'z:', z
        if ood:
            print 'out of domain'
        else:
            print 'y exp(x/y):', y*math.exp(x/y)
            print 'f = y exp(x/y) - z:', f
        # Out of domain.
        if ood:
            return None
        # Compute the gradient.
        Df = cvxopt.matrix([math.exp(x/y),
                            math.exp(x/y)*(1-x/y),
                            -1]).T
        if scaling is None:
            return f, Df
        # Compute the Hessian.
        H = math.exp(x/y)*cvxopt.matrix([
                [1.0/y, -x/y**2, 0.0],
                [-x/y**2, x**2/y**3, 0.0],
                [0.0, 0.0, 0.0],
            ])
        print 'H:'
        print H
        print
        return f, Df, scaling*H
