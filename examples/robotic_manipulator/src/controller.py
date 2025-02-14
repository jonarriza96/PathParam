import numpy as np
import casadi as cs
from scipy.linalg import block_diag
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from examples.robotic_manipulator.src.model import (
    robotic_manipulator,
    robotic_manipulator_spatial,
    robotic_manipulator_fullspatial,
    calculate_xidot,
)


def mpc_solver(T, N, x0):

    # solver
    ocp = AcadosOcp()

    # model
    ocp.model = robotic_manipulator(x0)

    # set dimensions
    ocp.dims.N = N

    # parameters
    n_param = ocp.model.p.shape[0]
    ocp.dims.np = n_param
    ocp.parameter_values = np.zeros(n_param)

    # set cost
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    x = ocp.model.x - ocp.model.p[:8]
    u = ocp.model.u  # - ocp.model.p[8:]

    q = np.diag(2 * [0] + 2 * [1e0] + 2 * [0] + 2 * [0])
    qe = np.diag(2 * [0] + 2 * [1e0] + 2 * [0] + 2 * [0])
    qu = np.diag(2 * [1e-4])

    ocp.model.cost_expr_ext_cost = x.T @ q @ x + u.T @ qu @ u
    ocp.model.cost_expr_ext_cost_e = 0  # x.T @ qe @ x

    # constraints
    # ocp.constraints.lbx = np.array([-3, -3, -3, -3])
    # ocp.constraints.ubx = np.array([3, 3, 3, 3])
    # ocp.constraints.idxbx = np.array([0, 1, 2, 3])
    # ocp.constraints.lbx = np.array([-np.pi, -np.pi])
    # ocp.constraints.ubx = np.array([np.pi, np.pi])
    # ocp.constraints.idxbx = np.array([4, 5])
    # ocp.constraints.lbu = np.array([-50, -50])
    # ocp.constraints.ubu = np.array([50, 50])
    # ocp.constraints.idxbu = np.array([0, 1])

    # set intial condition
    ocp.constraints.x0 = ocp.model.x0

    # set QP solver and integration
    ocp.solver_options.tf = T
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "ERK"
    # ocp.solver_options.sim_method_num_stages = 4
    # ocp.solver_options.sim_method_num_steps = 3

    # create solver
    acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")
    return acados_solver


def mpc_spatial_solver(T, N, x0, path):

    # solver
    ocp = AcadosOcp()

    # model
    ocp.model = robotic_manipulator_spatial(x0, path)

    # set dimensions
    ocp.dims.N = N

    # parameters
    n_param = ocp.model.p.shape[0]
    ocp.dims.np = n_param
    ocp.parameter_values = np.zeros(n_param)

    # set cost
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # ---------------------------------------------------------------------------- #
    v_profile = 0
    if not v_profile:
        x = ocp.model.x - ocp.model.p[:8]
        u = ocp.model.u  # - ocp.model.p[8:]

        q = np.diag(2 * [0] + 2 * [1e0] + 2 * [0] + 2 * [0])
        qe = np.diag(2 * [0] + 2 * [1e0] + 2 * [0] + 2 * [0])
        qu = np.diag(2 * [1e-2])

        ocp.model.cost_expr_ext_cost = x.T @ q @ x + u.T @ qu @ u
        ocp.model.cost_expr_ext_cost_e = 0  # x.T @ qe @ x
    else:
        xidot = calculate_xidot(x=ocp.model.x, path=path)
        xidot_ref = 0.05  # path["xi_dot"](ocp.model.p[2])
        u = ocp.model.u

        ocp.model.cost_expr_ext_cost = (xidot - xidot_ref) ** 2  # + 1e-6 * u.T @ u
        ocp.model.cost_expr_ext_cost_e = 0  # (xidot - xidot_ref) ** 2  # x.T @ qe @ x
    # ---------------------------------------------------------------------------- #
    # constraints
    # ocp.constraints.lbx = np.array([-3, -3, -3, -3])
    # ocp.constraints.ubx = np.array([3, 3, 3, 3])
    # ocp.constraints.idxbx = np.array([0, 1, 2, 3])
    # ocp.constraints.lbx = np.array([-np.pi, -np.pi])
    # ocp.constraints.ubx = np.array([np.pi, np.pi])
    # ocp.constraints.idxbx = np.array([4, 5])
    # ocp.constraints.lbu = np.array([-50, -50])
    # ocp.constraints.ubu = np.array([50, 50])
    # ocp.constraints.idxbu = np.array([0, 1])

    # set intial condition
    ocp.constraints.x0 = ocp.model.x0

    # set QP solver and integration
    ocp.solver_options.tf = T
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "ERK"
    # ocp.solver_options.sim_method_num_stages = 4
    # ocp.solver_options.sim_method_num_steps = 3

    # create solver
    acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")
    return acados_solver


def mpc_fullspatial_solver(T, N, x0, path):

    # solver
    ocp = AcadosOcp()

    # model
    ocp.model = robotic_manipulator_fullspatial(x0, path)

    # set dimensions
    ocp.dims.N = N

    # parameters
    n_param = ocp.model.p.shape[0]
    ocp.dims.np = n_param
    ocp.parameter_values = np.zeros(n_param)

    # set cost
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # ---------------------------------------------------------------------------- #
    x = ocp.model.x - ocp.model.p[:8]
    u = ocp.model.u  # - ocp.model.p[8:]

    q = np.diag([0, 1e-1] + [1e0, 1e-4] + 4 * [0])
    qu = np.diag(2 * [1e-4])
    qe = 0 * q

    ocp.model.cost_expr_ext_cost = x.T @ q @ x + u.T @ qu @ u
    ocp.model.cost_expr_ext_cost_e = x.T @ qe @ x
    # ---------------------------------------------------------------------------- #

    ocp.constraints.lbx = np.array([0])
    ocp.constraints.ubx = np.array([1])
    ocp.constraints.idxbx = np.array([0])

    # set intial condition
    ocp.constraints.x0 = ocp.model.x0

    # set QP solver and integration
    ocp.solver_options.tf = T
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "ERK"
    # ocp.solver_options.sim_method_num_stages = 4
    # ocp.solver_options.sim_method_num_steps = 3

    # create solver
    acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")
    return acados_solver
