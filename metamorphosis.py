from firedrake import *
from inner import solve_inner
from injection import injection
from petsc4py import PETSc
from termcolor import colored
import os

op2.init(log_level="WARNING")
parameters['assembly_cache']['enabled'] = False


class Metamorphosis:
    """ A class representing a metamorphosis problem. """

    def __init__(self, alphas, c, path):
        self.an, self.a0, self.a1, self.a2 = alphas
        self.c = c
        self.path = path

        if not os.path.exists(self.path):
            os.makedirs(path)

    def compute_deformation_energy(self, u):
        """ Computes the tri-Helmholtz norm corresponding to the
            first norm of the metamorphosis functional. """
        return assemble((self.an * u**2 + self.a0 * u.dx(0)**2
                       + self.a1 * u.dx(0).dx(0)**2
                       + self.a2 * u.dx(0).dx(0).dx(0)**2) * dx)


    def to_firedrake(self, V, L, u_spl):
        """ Convert a solution in spline space to a CG Firedrake function
            using the injection operator `L`. """

        _, dim_spl = L.getSize()
        fd_petsc = PETSc.Vec()
        fd_petsc.create()
        fd_petsc.setSizes(dim_spl)
        fd_petsc.setUp()

        fd_sol = Function(V)

        # convert PETSc object to Firedrake.Function
        L.multTranspose(u_spl, fd_petsc)
        with fd_sol.dat.vec as fd_vec:
            fd_petsc = fd_petsc.getValues(range(fd_petsc.getSize()))
            fd_vec.setValues(range(fd_vec.getSize()), fd_petsc)

        return fd_sol

    def compute_residual(self, u_sol, u_cg, L, V, bcs):
        """ Compute the residual given by the result of one Picard iteration
            of the Euler-Lagrange equation for the outer problem. """

        I_k = solve_inner(V, u_cg, bcs)

        u = TrialFunction(V)
        du = TestFunction(V)

        # Set up outer matrix problem
        A = (self.an * u*du + self.a0 * u.dx(0) * du.dx(0)
           + self.a1 * u.dx(0).dx(0) * du.dx(0).dx(0)
           + self.a2 * u.dx(0).dx(0).dx(0) * du.dx(0).dx(0).dx(0)
           + self.c * du * u * I_k.dx(0)**2) * dx
        F = - self.c * du * I_k.dx(1) * I_k.dx(0) * dx

        # assemble CG system
        A = assemble(A).M.handle
        F = assemble(F)

        # load vector
        LF = PETSc.Vec()
        LF.create()
        sp_dofs, _ = L.getSize()
        LF.setSizes(sp_dofs)
        LF.setUp()
        with F.dat.vec_ro as F_vec:
            L.mult(F_vec, LF)

        # stiffness matrix
        LA = L.matMult(A)
        LALT = LA.matTransposeMult(L)
        Lu = LF.duplicate()
        LALT.mult(u_sol, Lu)

        return (Lu - LF).norm()


    def solve_system(self, a, f, res, V, W, bcs):
        """ Solve the discretised, linearised Euler-Lagrange equation for the
            outer problem. """

        # compute injection
        L = injection(res, V, W)

        # assemble CG system
        A = assemble(a).M.handle
        F = assemble(f)

        # stiffness matrix
        LA = L.matMult(A)
        LALT = LA.matTransposeMult(L)

        # load vector
        LF = PETSc.Vec()
        LF.create()
        sp_dofs, _ = L.getSize()
        LF.setSizes(sp_dofs)
        LF.setUp()

        with F.dat.vec_ro as F_vec:
            L.mult(F_vec, LF)

        solution = LF.duplicate()

        ksp = PETSc.KSP().create()
        opts = PETSc.Options()
        opts['ksp_type'] = 'preonly'
        opts['pc_type'] = 'gamg'
        opts['mg_levels_ksp_type'] = 'chebyshev'
        opts['mg_levels_ksp_chebyshev_esteig'] = None
        opts['mg_levels_ksp_max_it'] = 5
        opts['mg_levels_pc_type'] = 'sor'
        opts['mg_levels_pc_sor_its'] = 1
        opts['pc_mg_cycles']='v'
        opts['pc_mg_type']='multiplicative'
        opts['pc_gamg_square_graph'] = 0
        opts['ksp_monitor'] = None
        ksp.setFromOptions()
        ksp.setOperators(LALT)
        ksp.setFromOptions()

        # solve
        ksp.solve(LF, solution)

        solution_cg = self.to_firedrake(V, L, solution)

        # compute residual
        res = self.compute_residual(solution, solution_cg, L, V, bcs)

        return solution, solution_cg, res


    def solve_full(self, res, length, degree, bcs, convergence_criteria,
                   u_guess, output=False):
        """ Solve the metamorphosis problem. """

        # Solver parameters
        r_tol    = convergence_criteria['r_tol']
        max_iter = convergence_criteria['max_iter']

        # Mesh and function spaces
        mesh = PeriodicIntervalMesh(res, length)
        mesh = ExtrudedMesh(mesh, layers=res, layer_height=length/res)
        e_t = FiniteElement("CG", interval, 1)
        e_x = FiniteElement("CG", interval, degree)
        e = TensorProductElement(e_x, e_t)
        V = FunctionSpace(mesh, e)

        ei = TensorProductElement(e_t, e_t)
        Vi = FunctionSpace(mesh, ei)

        mesh_ = UnitIntervalMesh(res)
        mesh_ = ExtrudedMesh(mesh_, res)
        W = FunctionSpace(mesh_, e)

        # Functions and initial guesses
        u_k = interpolate(u_guess, V)
        I_k = solve_inner(V, u_k, bcs)

        u  = TrialFunction(V)
        du = TestFunction(V)

        # Set up outer matrix problem
        A = (self.an * u*du + self.a0 * u.dx(0) * du.dx(0)
           + self.a1 * u.dx(0).dx(0) * du.dx(0).dx(0)
           + self.a2 * u.dx(0).dx(0).dx(0) * du.dx(0).dx(0).dx(0)) * dx
        F = lambda I_k: - self.c * du * (I_k.dx(1) + u * I_k.dx(0)) * I_k.dx(0) * dx
        u = Function(V)

        k = 0

        residual = float("inf")
        while (residual > r_tol and k < max_iter):
            k += 1
            #####
            # 1 #  update velocity
            #####
            # solve the outer problem using petsc4py
            usp, u, residual = self.solve_system(A, F(I_k), res, V, W, bcs)

            # update velocity using dampened Picard
            w = 0.2
            u_k = interpolate(w * u + (1 - w) * u, V)

            #####
            # 2 #  update image using the new velocity
            #####
            I = solve_inner(Vi, u_k, bcs)
            I_k = I

            #####
            # 3 #  dump residual
            #####

            r_str  = '=========> k = %d: residual = %s\n' % (k, str(residual))
            r_str += '                    norm   = %s' % str(norm(u))
            print colored(r_str, 'red')

            if output:
                File(path + '/u_' + str(k) + '.pvd').write(u_k)
                File(path + '/I_' + str(k) + '.pvd').write(I_k)

        return u_k, I_k


