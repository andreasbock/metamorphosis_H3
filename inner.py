from firedrake import *
import numpy as np
op2.init(log_level="WARNING")
parameters['assembly_cache']['enabled'] = False

def solve_inner(V, u, bcs):
    """
        V: function space to solve the problem on
        u: velocity of advection
        bcs: boundary conditions as `Expression`s
    """

    # BCs
    top    = DirichletBC(V, bcs[0], "top")
    bottom = DirichletBC(V, bcs[1], "bottom")
    bcs    = [top, bottom]

    # Functions
    I  = Function(V)
    dI = TestFunction(V)

    # Construct and solve problem
    solver_parameters = {'ksp_type': 'preonly',
                         'pc_type': 'lu'}
    solver_parameters = {'ksp_type': 'cg',
                         'pc_type': 'lu'}
    F = ((dI.dx(1) + u * dI.dx(0)) * (I.dx(1) + u * I.dx(0))) * dx
    solve(F == 0, I, bcs=bcs, solver_parameters=solver_parameters)
    return I


def solve_inner_test(res, degree, u, bcs, analytical, suff, path):

    mesh = PeriodicUnitIntervalMesh(res)
    mesh = ExtrudedMesh(mesh, res)
    e = FiniteElement("CG", interval, degree)
    e = TensorProductElement(e, e)
    V = FunctionSpace(mesh, e)

    u = interpolate(u, V)
    I = solve_inner(V, u, bcs)
    File('imgs/I_' + str(res) + '_' + suff + '.pvd').write(I)

    analytical = interpolate(analytical, V)
    File('analytical.pvd').write(analytical)

    l2 = errornorm(I, analytical)

    diff = interpolate(I - analytical, V)
    engy = sqrt(assemble(((diff.dx(1) + u * diff.dx(0))**2) * dx))

    return (l2, engy)

