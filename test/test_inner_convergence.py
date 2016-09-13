from __future__ import division
from inner import solve_inner_test
import os
import numpy as np
from firedrake import *
import matplotlib.pyplot as plt

path = 'imgs'


def dump_pdf(res, error, idx):
    fig, ax = plt.subplots()
    l2, h1 = zip(*error)
    print l2
    ymin = min(min(l2), min(h1))
    ymax = max(max(l2), max(h1))
    l2 = plt.loglog(res, l2, 'o-', color='orange', label='$L^2$')
    h1 = plt.loglog(res, h1, 'x-', color='blue', label='Energy norm')
    ax.grid(True)
    ax.set_xlabel(r'Number of elements ($h$)', fontsize=12)
    ax.set_ylabel(r'Error', fontsize=12)
    ax.set_ylim([ymin, ymax])
    ax.set_xlim([res[0], res[-1]])
    #plt.xticks(np.arange(min(res), max(res)+1, 14), np.arange(min(res), max(res)+1, 14))
    #plt.yticks(np.arange(ymin, ymax, 0.3), np.around(np.arange(ymin, ymax, 0.3), 1))
    plt.legend(loc=1)
    plt.savefig(path + '/conv_test_' + str(idx) + '.pdf')

#def test_convergence_0():

    #degree = 1
    #bcs = map(Expression, ["exp(1)*exp(-exp(x[0]-0.5))", "exp(-exp(x[0]-0.5))"])
    ##bcs = [Expression("exp(1)*exp(-exp(x[0]-0.5))"), Expression("exp(-exp(x[0]-0.5))")]
    #u = Expression("exp(-x[0]+0.5)")
    #analytical = Expression("exp(-exp(x[0]-0.5))*exp(x[1])")

    #res = [2**i for i in range(3, 7)]
    #error = [solve_inner_test(r, degree, u, bcs, analytical) for r in res]
    #convergence_rate = np.array([np.log(error[i]/error[i+1])/np.log(res[i+1]/res[i])
                                 #for i in range(len(res)-1)])

    #print "Achieved convergence rates: %s" % convergence_rate


#def test_convergence_1():

    #degree = 1
    #bcs = map(Expression, ["exp(x[0])"] * 2)
    #u = Expression("cos(pi*x[1])")
    #analytical = Expression("exp(x[0])*exp(-(sin(pi*x[1]))/pi)")

    #res = [2**i for i in range(3, 7)]
    #error = [solve_inner_test(r, degree, u, bcs, analytical) for r in res]
    #convergence_rate = np.array([np.log(error[i]/error[i+1])/np.log(res[i+1]/res[i])
                                 #for i in range(len(res)-1)])

    #print "Achieved convergence rates: %s" % convergence_rate


def test_convergence_0():

    suffix = "0"
    degree = 1
    bcs = map(Expression, ["exp(-100*pow(x[0]-0.5,2))"] * 2)
    u = Expression("1 - 2*x[1]")
    analytical = Expression("exp(-100*pow(x[0] - x[1] * (1-x[1]) -0.5,2))")

    res = [2**i for i in range(3, 8)]
    error = [solve_inner_test(r, degree, u, bcs, analytical, suffix, path) for r in res]
    dump_pdf(res, error, 0)
    #convergence_rate = np.array([np.log(error[i]/error[i+1])/np.log(res[i+1]/res[i])
                                 #for i in range(len(res)-1)])
    #print "Achieved convergence rates: %s" % convergence_rate


def test_convergence_1():

    suffix = "1"
    degree = 1
    #bcs = map(Expression, ["exp(-pow(x[0]-0.4,2)/0.008)", "exp(-pow(x[0]-0.6,2)/0.008)"] )
    bcs = map(Expression, ["exp(-pow(x[0]-0.3,2)/0.008)", "exp(-pow(x[0]-0.7,2)/0.008)"] )
    analytical = Expression("exp(-pow(x[0] - 0.4 * (1-x[1]) - 0.3,2)/0.008)")
    u = Expression("-0.4")

    res = [2**i for i in range(3, 8)]
    error = [solve_inner_test(r, degree, u, bcs, analytical, suffix, path) for r in res]
    dump_pdf(res, error, 1)
    #convergence_rate = np.array([np.log(error[i]/error[i+1])/np.log(res[i+1]/res[i])
                                 #for i in range(len(res)-1)])
    #print "Achieved convergence rates: %s" % convergence_rate


def test_convergence_2():

    suffix = "2"
    degree = 1
    bcs = map(Expression, ["exp(-25*pow(-4.3*x[0]+5.16*pow(x[0],3)+0.5+1,2))", "exp(-25*pow(-4.3*x[0]+5.16*pow(x[0],3)+0.5,2))"])
    analytical = Expression("exp(-25*pow(-4.3*x[0]+5.16*pow(x[0],3)+0.5+x[1],2))")
    u = Expression("-5/(-21.5+77.4*pow(x[0],2))")

    res = [2**i for i in range(3, 8)]
    error = [solve_inner_test(r, degree, u, bcs, analytical, suffix, path) for r in res]
    dump_pdf(res, error, 2)
    #convergence_rate = np.array([np.log(error[i]/error[i+1])/np.log(res[i+1]/res[i])
                                 #for i in range(len(res)-1)])
    #print "Achieved convergence rates: %s" % convergence_rate


if __name__ == '__main__':
    if not os.path.exists(path):
        os.makedirs(path)

    test_convergence_0()
    test_convergence_1()
    test_convergence_2()
