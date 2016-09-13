from __future__ import division
from firedrake import *
import numpy as np
import itertools
import matplotlib.pyplot as plt
import scipy.sparse as sps
from pyop2 import op2
from petsc4py import PETSc
from Bspline import Bspline
import pdb
p = pdb.set_trace


def cg_index(i, layers):
    """ Given a CG1-CG1 index, returns the CG3 x CG-t_deg node number. """
    # TODO: if out of bounds!
    return (i // 4) * 12 + (i % 4) * 1 # change last to degree in time


def adj_nodes(i, V):
    """ Return the Lagrange nodes around node `i` that we need to tabulate
        our b-spline at. """

    deg_h, deg_v = V.ufl_element().degree()
    layers = V.mesh().topology.layers
    i = cg_index(i, layers)

    dofs_v = layers * deg_v # number of vertices vertically

    # Expanding horizontally to cover a total of 4 cells (edges, really)
    # dofs_v: how far we need to move left-right horizontally once
    # deg_h * 2: how many times we need to move left-right
    h_step = dofs_v * deg_h * 2
    n_nodes = dofs_v
    nodes = np.arange(i - h_step, i + h_step + 1, n_nodes)

    # Use offset to compute three rows of 4 cells
    offset = V.cell_node_map().offset[0]
    nodes = np.vstack([nodes + offset, nodes, nodes - offset])

    return nodes % V.dof_count

def adj_coords(nodes, res, V):
    """ Return the coordinates around node `i` that we need to tabulate
        our b-spline at. """

    deg_h, deg_v = V.ufl_element().degree()

    # extract coordinates at the vertices
    coords = V.mesh().coordinates.dat.data

    # compute the CG points corresponding to the node numbers
    xs = np.linspace(0, 1, res * deg_h + 1)
    ts = np.linspace(0, 1, res * deg_v + 1)

    cg_coords = np.zeros((len(xs), len(ts)), dtype=object)
    for i, x in enumerate(xs):
        for j, t in enumerate(ts):
            cg_coords[i,j] = (x, t)

    # mapping from global node to coordinates
    cg_coords = cg_coords.flatten()
    f = lambda i: cg_coords[i]

    return f(nodes)

def injection(res, V, W):
    """ Compute the injection operator. """

    # tabulate in the spatial dimension
    spl_deg = 3
    lagrange_points_x = np.linspace(0, 1, 4 * spl_deg + 1)
    knots = lagrange_points_x[::spl_deg]
    phi_x = Bspline(knots, spl_deg) # cubic in space

    xs = np.array([phi_x(x) for x in lagrange_points_x]).flatten()
    # tabulate in the temporal dimension
    def phi_t(t):
        t2, t1, t0 = np.linspace(1, 0, 3)
        if t >= t1: return (t2 - t)/(t2 - t1)
        else: return (t - t0)/(t1 - t0)

    _, deg_v = V.ufl_element().degree()
    lagrange_points_t = np.linspace(0, 1, 2 * deg_v + 1)
    ts = np.array([phi_t(t) for t in lagrange_points_t])

    # TODO: use tensorfun
    vals = np.array([[x*t for x in xs] for t in ts]).flatten()

    # compute number of b-spline DOFs in the periodic mesh
    l = V.mesh().topology.layers
    sp_dofs = l * (l - 1)
    L = PETSc.Mat()
    L.create()
    L.setSizes((sp_dofs, V.dof_count))
    L.setType('aij')
    L.setUp()

    # loop over each dof to populate L
    for i in range(sp_dofs):
        js = adj_nodes(i, V).flatten()
        for j, val in zip(js, vals):
            if j not in range(V.dof_count):
                print "j = " + str(j) +  " not in thing!"
                exit()
            L.setValue(i, j, val)
    L.assemble()

    return L

