from firedrake import *
from metamorphosis import Metamorphosis
import inspect
import os

def run_test(alphas, c, bcs, u_guess, path):
    # parameters
    res    = 20
    length = 1.0
    degree = 3  # cubic spline


    # convergence criteria
    r_tol = 1e-07
    max_iter = 20
    convergence_criteria = {'r_tol' : r_tol,
                            'max_iter' : max_iter}
    output = 0

    mm = Metamorphosis(alphas, c, path)
    u, I = mm.solve_full(res, length, degree, bcs, convergence_criteria,
                         u_guess, output)

    File(path + '/I.pvd').write(I)
    File(path + '/u.pvd').write(u)


####################################
# Test 0: a transportation of mass #
####################################

def test_0(path):
    alphas = [0.000000000000001] * 4
    c  = 0.000001

    top = Expression("exp(-pow(x[0]-0.3,2)/0.008)")
    bot = Expression("exp(-pow(x[0]-0.7,2)/0.008)")
    bcs = [top, bot]
    u_guess = Expression("0.2")

    subdir = path + '/' + inspect.stack()[0][3]
    run_test(alphas, c, bcs, u_guess, subdir)

###################
# Test 1: a split #
###################

# Transportation: LEFT
def test_1_left(path):
    alphas = [0.000000000000001] * 4
    c  = 0.000001 # <--- LEFT

    top = Expression("exp(-pow(x[0]-mean0,2)/std0) + exp(-pow(x[0]-mean1,2)/std1)",
                     std0=.01, mean0=0.2, std1=.01, mean1=0.8)
    bot = Expression("exp(-pow(x[0]-mean,2)/std)", std=.01, mean=0.4)
    bcs = [top, bot]
    u_guess = Expression("0.2")

    subdir = path + '/' + inspect.stack()[0][3]
    run_test(alphas, c, bcs, u_guess, subdir)


# Transportation: RIGHT
def test_1_right(path):
    alphas = [0.000000000000001] * 4
    c  = 0.0000005 # <--- RIGHT

    top = Expression("exp(-pow(x[0]-mean0,2)/std0) + exp(-pow(x[0]-mean1,2)/std1)",
                     std0=.01, mean0=0.2, std1=.01, mean1=0.8)
    bot = Expression("exp(-pow(x[0]-mean,2)/std)", std=.01, mean=0.4)
    bcs = [top, bot]
    u_guess = Expression("0.2")

    subdir = path + '/' + inspect.stack()[0][3]
    run_test(alphas, c, bcs, u_guess, subdir)

##########################################
# Test 2: deformation and transportation #
##########################################

def test_2(path):
    alphas = [0.000000000000001] * 4
    c  = 0.000001

    top = Expression("exp(-pow(x[0]-mean,2)/std)", std=.01, mean=0.8)
    bot = Expression("exp(-pow(x[0]-mean,2)/std)", std=.05, mean=0.4)
    bcs = [top, bot]
    u_guess = Expression("-0.6") # or 0.1 works too!

    subdir = path + '/' + inspect.stack()[0][3]
    run_test(alphas, c, bcs, u_guess, subdir)


if __name__ == '__main__':

    path = 'imgs'
    if not os.path.exists(path):
        os.makedirs(path)

    test_0(path)
    test_1_left(path)
    test_1_right(path)
    test_2(path)
