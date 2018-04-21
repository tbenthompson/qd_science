import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

import tectosaur
from tectosaur.util.timer import Timer
from tectosaur.mesh.combined_mesh import CombinedMesh
from tectosaur.ops.mass_op import MassOp
from tectosaur.ops.sparse_integral_op import SparseIntegralOp, FMMFarfieldBuilder
from tectosaur.constraint_builders import free_edge_constraints, continuity_constraints
from tectosaur.constraints import build_constraint_matrix
from tectosaur.util.geometry import unscaled_normals
import tectosaur_topo.solve

import cppimport.import_hook
import qd_newton

def plot_fields(m, field, levels = None, cmap = 'seismic', symmetric_scale = False, ds = None, figsize = None):
    field_reshape = field.reshape(m.tris.shape[0],3,-1)
    n_fields = field_reshape.shape[2]
    if figsize is None:
        figsize = (6 * n_fields,5)
    plt.figure(figsize = figsize)
    if ds is None:
        ds = range(n_fields)
    for d in ds:
        plt.subplot(1, n_fields, d + 1)
        pt_field = np.empty((m.pts.shape[0], n_fields))
        pt_field[m.tris] = field_reshape
        f = pt_field[:,d]
        f_levels = levels
        if f_levels is None:
            min_f = np.min(f)
            max_f = np.max(f)
            scale_f = np.max(np.abs(f))
            if scale_f == 0.0:
                scale_f = 1.0
            min_f -= 1e-13 * scale_f
            max_f += 1e-13 * scale_f
            if symmetric_scale:
                min_f = -max_f
            f_levels = np.linspace(min_f, max_f, 21)
        cntf = plt.tricontourf(m.pts[:,0], m.pts[:,2], m.tris, f, cmap = cmap, levels = f_levels, extend = 'both')
        #plt.tricontour(m.pts[:,0], m.pts[:,2], m.tris, f, levels = levels, extend = 'both', linestyles = 'solid', linewidths = 0.75, colors = '#333333')
        plt.colorbar(cntf)
        #plt.axis('equal')
    plt.show()


def make_integral_op(m, k_name, k_params, cfg, name1, name2):
    if cfg['use_fmm']:
        farfield = FMMFarfieldBuilder(
            cfg['fmm_order'], cfg['fmm_mac'], cfg['pts_per_cell']
        )
    else:
        farfield = None
    return SparseIntegralOp(
        cfg['quad_vertadj_order'], cfg['quad_far_order'],
        cfg['quad_near_order'], cfg['quad_near_threshold'],
        k_name, k_params, m.pts, m.tris, cfg['float_type'],
        farfield_op_type = farfield,
        obs_subset = m.get_tri_idxs(name1),
        src_subset = m.get_tri_idxs(name2)
    )

def make_mass_op(m, cfg):
    return MassOp(cfg['quad_mass_order'], m.pts, m.tris)

def get_slip_to_traction(qdm, qd_cfg):
    tectosaur.logger.setLevel(qd_cfg['tectosaur_cfg']['log_level'])
    tectosaur_topo.logger.setLevel(qd_cfg['tectosaur_cfg']['log_level'])
    cs2 = continuity_constraints(qdm.m.get_tris('fault'), np.array([]))
    cs2.extend(free_edge_constraints(qdm.m.get_tris('fault')))
    cm2, c_rhs2 = build_constraint_matrix(cs2, qdm.m.n_dofs('fault'))
    hypersingular_op = make_integral_op(qdm.m, 'elasticH3', [qd_cfg['sm'], qd_cfg['pr']], qd_cfg['tectosaur_cfg'], 'fault', 'fault')
    traction_mass_op = make_mass_op(qdm.m, qd_cfg['tectosaur_cfg'])
    constrained_traction_mass_op = cm2.T.dot(traction_mass_op.mat.dot(cm2))
    def slip_to_traction(slip):
        t = Timer()
        rhs = hypersingular_op.dot(slip)
        t.report('H.dot')
        #return spsolve(traction_mass_op.mat, rhs)
        out = cm2.dot(spsolve(constrained_traction_mass_op, cm2.T.dot(rhs)))
        return out
        #t.report('spsolve')
        #return out
        #out_vec = out.reshape((-1,3))
        #out_vec[:,2] = 0.0
        #return out_vec.reshape(-1)
    return slip_to_traction

def get_traction_to_slip(qdm, qd_cfg):
    tectosaur.logger.setLevel(qd_cfg['tectosaur_cfg']['log_level'])
    tectosaur_topo.logger.setLevel(qd_cfg['tectosaur_cfg']['log_level'])
    cs2 = continuity_constraints(qdm.m.get_tris('fault'), np.array([]))
    cs2.extend(free_edge_constraints(qdm.m.get_tris('fault')))
    cm2, c_rhs2 = build_constraint_matrix(cs2, qdm.m.n_dofs('fault'))
    hypersingular_op = make_integral_op(qdm.m, 'elasticH3', [qd_cfg['sm'], qd_cfg['pr']], qd_cfg['tectosaur_cfg'], 'fault', 'fault')
    traction_mass_op = make_mass_op(qdm.m, qd_cfg['tectosaur_cfg'])
    #constrained_traction_mass_op = cm2.T.dot(traction_mass_op.mat.dot(cm2))
    def traction_to_slip(traction):
        rhs = traction_mass_op.dot(traction)
        out = tectosaur_topo.solve.iterative_solve(
            hypersingular_op, cm2, rhs, lambda x: x, dict(solver_tol = 1e-8)
        )
        return out
    return traction_to_slip

class QDMeshData:
    def __init__(self, m):
        self.m = CombinedMesh.from_named_pieces([('fault', m)])

        self.unscaled_tri_normals = unscaled_normals(self.m.pts[self.m.tris])
        self.tri_normals = self.unscaled_tri_normals / np.linalg.norm(self.unscaled_tri_normals, axis = 1)[:, np.newaxis]

        cs = free_edge_constraints(self.m.get_tris('fault'))
        cm, c_rhs = build_constraint_matrix(cs, self.m.n_dofs('fault'))

        constrained_slip = np.ones(cm.shape[1])
        self.ones_interior = cm.dot(constrained_slip)
        self.field_100_interior = self.ones_interior.copy()
        self.field_100_interior.reshape(-1,3)[:,1] = 0.0
        self.field_100_interior.reshape(-1,3)[:,2] = 0.0

        self.field_100 = self.field_100_interior.copy()
        self.field_100.reshape(-1,3)[:,0] = 1.0
        self.field_100_edges = self.field_100 - self.field_100_interior

def separate_slip_state(y):
    n_total_dofs = y.shape[0]
    n_slip_dofs = n_total_dofs // 4 * 3
    return y[:n_slip_dofs], y[n_slip_dofs:]

def get_plate_motion(qdm, qd_cfg, t):
    return t * qd_cfg['plate_rate'] * qdm.field_100

def get_slip_deficit(qdm, qd_cfg, t, slip):
    out = qdm.ones_interior * (get_plate_motion(qdm, qd_cfg, t).reshape(-1) - slip)
    np.testing.assert_almost_equal(qdm.field_100_edges * out, 0.0)
    return out

def rate_state_solve(qdm, qd_cfg, traction, state):
    V = np.empty_like(qdm.field_100)
    qd_newton.rate_state_solver(
        qdm.tri_normals, traction, state, V,
        qd_cfg['a'], qd_cfg['eta'], qd_cfg['V0'],
        qd_cfg['additional_normal_stress'],
        1e-12, 50
    )
    return qdm.field_100_edges * qd_cfg['plate_rate'] + qdm.ones_interior * V

# State evolution law -- aging law.
def aging_law(qd_cfg, V, state):
    return (qd_cfg['b'] * qd_cfg['V0'] / qd_cfg['Dc']) * (
        np.exp((qd_cfg['f0'] - state) / qd_cfg['b']) - (V / qd_cfg['V0'])
    )

def state_evolution(qdm, qd_cfg, V, state):
    V_mag = np.linalg.norm(V.reshape(-1,3), axis = 1)
    qdm.max_V = np.max(V_mag)
    return aging_law(qd_cfg, V_mag, state)

def solve_for_full_state(qdm, qd_cfg, slip_to_traction, t, y):
    tm = Timer()
    slip, state = separate_slip_state(y)
    tm.report('sep')
    slip_deficit = get_slip_deficit(qdm, qd_cfg, t, slip)
    tm.report('slip deficit')

    traction = slip_to_traction(slip_deficit)
    tm.report('slip_to_traction')
    V = rate_state_solve(qdm, qd_cfg, traction, state)
    tm.report('rate_state_solve')
    dstatedt = state_evolution(qdm, qd_cfg, V, state)
    tm.report('state_evolution')
    return slip, slip_deficit, state, traction, V, dstatedt

def make_qd_derivs(qdm, qd_cfg, slip_to_traction):
    def qd_derivs(t, y):
        slip, slip_deficit, state, traction, V, dstatedt = solve_for_full_state(
            qdm, qd_cfg, slip_to_traction, t, y
        )
        #print('Vmax', np.max(V.reshape((-1,3)), axis = 0))
        #TODO: CHECK THE DIMENSIONS ON V AND dstatedt
        return np.concatenate((V, dstatedt))
    return qd_derivs

def plot_setting(t, y, qdm, qd_cfg, slip_to_traction):
    slip, slip_deficit, state, traction, V, dstatedt = solve_for_full_state(
        qdm, qd_cfg, slip_to_traction, t, y
    )
    #print('slip')
    #plot_signs(slip)
    #plot_fields(np.log10(np.abs(slip) + 1e-40))
    #print('deficit')
    #plot_signs(slip_deficit)
    #plot_fields(np.log10(np.abs(slip_deficit) + 1e-40))
    print('slip')
    plot_fields(qdm.m, slip)
    print('V')
    #plot_signs(V)
    plot_fields(qdm.m, np.log10(np.abs(V) + 1e-40))
    print('traction')
    min_trac = 0.9 * np.max(traction)
    max_trac = np.max(traction)
    plot_fields(qdm.m, traction, levels = np.linspace(min_trac, max_trac, 20))
    print('state')
    plot_fields(qdm.m, state)
