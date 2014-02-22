"""
Find an optimal compromise between two distributions.

"""
from __future__ import division, print_function, absolute_import
import numpy as np
import cvxpy
from cvxpy import square, kl_div, norm1

n = 4
target_distn = np.array([0.1, 0.2, 0.3, 0.4])
uniform_distn = np.array([0.25, 0.25, 0.25, 0.25])

# parameters with the simplex constraint
#x = cvxpy.Variable(n)
x0 = cvxpy.Variable(1)
x1 = cvxpy.Variable(1)
x2 = cvxpy.Variable(1)
x3 = cvxpy.Variable(1)
#constraints = [0 <= x, sum(x)==1]
constraints = [
        0 <= x0,
        0 <= x1,
        0 <= x2,
        0 <= x3,
        x0 + x1 + x2 + x3 == 1]

# try to match the target distribution with a weird loss
#error = square(sum([kl_div(a, b) for a, b in zip(target_distn, x)]))
error = square(
        kl_div(target_distn[0], x0) +
        kl_div(target_distn[1], x1) +
        kl_div(target_distn[2], x2) +
        kl_div(target_distn[3], x3))

# regularize using the total variation distance to the uniform distribution
penalty = 0.5 * (
        cvxpy.abs(x0 - uniform_distn[0]) +
        cvxpy.abs(x1 - uniform_distn[1]) +
        cvxpy.abs(x2 - uniform_distn[2]) +
        cvxpy.abs(x3 - uniform_distn[3]))

#objective = cvxpy.Minimize(error + penalty)
#objective = cvxpy.Minimize(error)
objective = cvxpy.Minimize(penalty)

p = cvxpy.Problem(objective, constraints)
result = p.solve(verbose=True)
print(result)
print(dir(result))
#print(result.x)
#print(result.x0)
#print(result.x1)
#print(result.x2)
#print(result.x3)

#for w in x.value:
    #print(w)

