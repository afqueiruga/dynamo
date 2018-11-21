import dolfin as dlfn
# physical ids
"""
surfaces:
    1   interior spherical surface
    2   exterior surface
volumes:
    1   interior material volume
    2   exterior vacuous volume
"""
intId = 1
extId = 2
intrfcId = 1
bndryId = 2
#------------------------------------------------------------------------------#
# time stepping
#------------------------------------------------------------------------------#
eta = 1.0
dt = 1e-2
n_steps = 20
t_end = 1.0
#------------------------------------------------------------------------------#
# import initial mesh
#------------------------------------------------------------------------------#
mesh_name = "../meshes/sphereInCube"
mesh = dlfn.Mesh(mesh_name + ".xml" )
dim = mesh.geometry().dim()
# load facet and cell ids
cellMeshFun = dlfn.MeshFunctionSizet(mesh,
                                     mesh_name + "_physical_region.xml")
facetMeshFun = dlfn.MeshFunctionSizet(mesh,
                                      mesh_name + "_facet_region.xml")
#------------------------------------------------------------------------------#
# function spaces, test/trial functions, ...
#------------------------------------------------------------------------------#
# finite element spaces
P1Curl = dlfn.FiniteElement("N1curl", mesh.ufl_cell(), 1)
P1 = dlfn.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
# function spaces
Wh = dlfn.FunctionSpace(mesh, P1Curl * P1)
# test and trial functions
A, phi = dlfn.TrialFunctions(Wh)
B, psi = dlfn.TestFunctions(Wh)
# measures and normal vectors
dS = dlfn.Measure("dS", mesh, subdomain_data=facetMeshFun)
dA = dlfn.Measure("ds", mesh, subdomain_data=facetMeshFun)
dV = dlfn.Measure("dx", mesh, subdomain_data=cellMeshFun)
normal = dlfn.FacetNormal(mesh)
#------------------------------------------------------------------------------#
# solution functions
#------------------------------------------------------------------------------#
sol = dlfn.Function(Wh)
sol0 = dlfn.Function(Wh)
sol_A = sol.sub(0)
sol_A0 = sol0.sub(0)
#------------------------------------------------------------------------------#
# weak forms and assembly
#------------------------------------------------------------------------------#
# linear forms in interior domain
from dolfin import curl, inner, dot, grad
a =   dlfn.Constant(1. / dt) * dot(A, B) * dV(intId) \
    + dlfn.Constant(0.5 * eta) * inner(curl(A), curl(B)) * dV(intId) \
    + inner(curl(A), curl(B)) * dV(extId) \
    + dot(A, grad(psi)) * dV \
    + dot(grad(phi), B) * dV
l =   dlfn.Constant(1. / dt) * dot(sol_A0, B) * dV(intId) \
    - dlfn.Constant(0.5 * eta) * inner(curl(sol_A0), curl(B)) * dV(intId)
# boundary condition
bcA = dlfn.DirichletBC(Wh.sub(0), dlfn.Constant((0.0, 0.0, 0.0)),
                       facetMeshFun, bndryId)
#------------------------------------------------------------------------------#
# linear solver
#------------------------------------------------------------------------------#
problem = dlfn.LinearVariationalProblem(a, l, sol, bcs=bcA)
solver = dlfn.LinearVariationalSolver(problem)
#------------------------------------------------------------------------------#
# initial conditions
#------------------------------------------------------------------------------#
class ExactVectorPotential(dlfn.Expression):
    def __init__(self, **kwargs):
        # user input check
        assert isinstance(kwargs["cell_data"], dlfn.MeshFunctionSizet)
        self._cell_data = kwargs["cell_data"]
        assert isinstance(kwargs["interior_id"], int)
        assert isinstance(kwargs["exterior_id"], int)
        self._interior_id = kwargs["interior_id"]
        self._exterior_id = kwargs["exterior_id"]
        self._dim = self._cell_data.mesh().topology().dim()
        if self._dim != 3:
            raise NotImplementedError()

    def value_shape(self):
        return (3,)

    def eval_cell(self, value, x, ufl_cell):
        # assign value depending on domain
        if self._cell_data[ufl_cell.index] == self._interior_id:
            from math import sqrt
            r = sqrt(x[0]**2 + x[1]**2 + x[2]**2)
            value[0] = - x[1] * (1. - r)
            value[1] = x[0] * (1. - r)
            value[2] = 0.
        elif self._cell_data[ufl_cell.index] == self._exterior_id:
            value[0] = 0.
            value[1] = 0.
            value[2] = 0.
        else:
            raise ValueError()
tmp = ExactVectorPotential(interior_id = intId, exterior_id = extId,
                           cell_data = cellMeshFun, degree = 2)
Vh = dlfn.FunctionSpace(mesh, P1Curl)
auxVh = dlfn.FunctionSpace(mesh, P1)
assigner = dlfn.FunctionAssigner(Wh, [Vh, auxVh])
A0 = dlfn.project(tmp, Vh)
phi0 = dlfn.project(dlfn.Constant(0.), auxVh)
assigner.assign(sol0, [A0, phi0])
#------------------------------------------------------------------------------#
# output preparation
#------------------------------------------------------------------------------#
# DG spaces for projection
P1Grad = dlfn.VectorElement("DG", mesh.ufl_cell(), 0)
H1Grad= dlfn.FunctionSpace(mesh, P1Grad)
# output files
pvd_A = dlfn.File("solution-A.pvd")
pvd_curlA = dlfn.File("solution-curlA.pvd")
#------------------------------------------------------------------------------#
# time loop
#------------------------------------------------------------------------------#
step = 0
time = 0.0
# write initial condition
dlfn.File("initial-A.pvd") << sol_A0
while time < t_end and step < n_steps:
    print "Iteration: {:08d}, ".format(step), "time = {0:10.5f},".format(time),\
            " time step = {0:5.4e}".format(dt)
    # solve linear system
    solver.solve()
    # time update
    time += dt
    step += 1
    # curl projection
    curl_A = dlfn.project(curl(sol_A), H1Grad)
    # write output
    pvd_A << (sol_A, time)
    pvd_curlA << (curl_A, time)
    # update solutions for next iteration
    sol0.assign(sol)