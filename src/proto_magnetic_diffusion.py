import dolfin as dlfn
from afqsfenicsutil.my_restriction_map  import *
from exact_solutions import ExactVectorPotential
from mesh_function_transfer import transferMeshFunToSubMesh
# wrapper for csr-matrix
def getScipySparseMatrix(mat):
    assert isinstance(mat, dlfn.la.Matrix)
    from scipy.sparse import csr_matrix
    m, n = mat.size(0), mat.size(1)
    indptr, indices, data = dlfn.as_backend_type(mat).mat()\
                            .getValuesCSR()
    return csr_matrix((data, indices, indptr), shape=(m, n))
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
facetIds = [intrfcId, bndryId]
subIds = [intId, extId]
#------------------------------------------------------------------------------#
# time stepping
#------------------------------------------------------------------------------#
eta = 1.0
dt = 1e-3
n_steps = 1000
t_end = 1.0
#------------------------------------------------------------------------------#
# import initial mesh
#------------------------------------------------------------------------------#
mesh_name = "../meshes/sphereInCube"
mesh = dlfn.Mesh(mesh_name + ".xml" )
dim = mesh.geometry().dim()
ncells = mesh.num_cells()
# load facet and cell ids
cellMeshFun = dlfn.MeshFunctionSizet(mesh,
                                     mesh_name + "_physical_region.xml")
facetMeshFun = dlfn.MeshFunctionSizet(mesh,
                                     mesh_name + "_facet_region.xml")
#------------------------------------------------------------------------------#
# sub meshes
#------------------------------------------------------------------------------#
# create submeshes
subMeshes = {intId: dlfn.SubMesh(mesh, cellMeshFun, intId),
             extId: dlfn.SubMesh(mesh, cellMeshFun, extId)}
# transfer mesh functions
facetSubMeshFuns = dict()
cellSubMeshFuns = dict()
for i in subIds:
    facetSubMeshFuns[i] = transferMeshFunToSubMesh(subMeshes[i], facetMeshFun)
    cellSubMeshFuns[i] = transferMeshFunToSubMesh(subMeshes[i], cellMeshFun)
#------------------------------------------------------------------------------#
# function spaces, test/trial functions, ...
#------------------------------------------------------------------------------#
# finite element spaces
P1Curl = dlfn.FiniteElement("N1curl", mesh.ufl_cell(), 1)
P1 = dlfn.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
# function spaces
intHCurl = dlfn.FunctionSpace(subMeshes[intId], P1Curl)
intH1 = dlfn.FunctionSpace(subMeshes[intId], P1)
extH1 = dlfn.FunctionSpace(subMeshes[extId], P1)
# size of function spaces
m = intHCurl.dim()
nn = intH1.dim()
n = extH1.dim()
# test and trial functions
A = dlfn.TrialFunction(intHCurl)
B = dlfn.TestFunction(intHCurl)
u = dlfn.TrialFunction(intH1)
phi = dlfn.TrialFunction(extH1)
psi = dlfn.TestFunction(extH1)
# measures and normal vectors
dA, dV, normals = dict(), dict(), dict()
for i in subIds:
    dA[i] = dlfn.Measure("ds", subMeshes[i], subdomain_data=facetSubMeshFuns[i])
    dV[i] = dlfn.Measure("dx", subMeshes[i])
    normals[i] = dlfn.FacetNormal(subMeshes[i])
intV =  dlfn.assemble(dlfn.Constant(1.0) * dV[intId])
extV =  dlfn.assemble(dlfn.Constant(1.0) * dV[extId])
#------------------------------------------------------------------------------#
# solution functions
#------------------------------------------------------------------------------#
sol_A = dlfn.Function(intHCurl)
sol_A0 = dlfn.Function(intHCurl)
sol_phi = dlfn.Function(extH1)
#------------------------------------------------------------------------------#
# weak forms and assembly in interior
#------------------------------------------------------------------------------#
# linear forms in interior domain
from dolfin import curl, inner, dot, grad
id_int = dlfn.Constant(1. / dt) * dot(A, B) * dV[intId]
a_int = dlfn.Constant(0.5 * eta) * inner(curl(A), curl(B)) * dV[intId] 
b_int = u * dot(normals[intId], curl(B)) * dA[intId](intrfcId)
l_int = dlfn.Constant(1. / dt) * dot(sol_A0, B) * dV[intId] \
        - dlfn.Constant(0.5 * eta) * inner(curl(sol_A0), curl(B)) * dV[intId]
# assemble (0,0)-block
matA = dlfn.assemble(a_int)
matM = dlfn.assemble(id_int)
# assemble (0,1)-block
matB = dlfn.assemble(b_int)
#------------------------------------------------------------------------------#
# weak forms and assembly in exterior
#------------------------------------------------------------------------------#
# linear forms in exterior domain
a_ext = inner(grad(phi), grad(psi)) * dV[extId]
# apply boundary condition
bc = dlfn.DirichletBC(extH1, dlfn.Constant(0.),
                      facetSubMeshFuns[extId], bndryId)
# assemble (2,2)-block
matC = dlfn.assemble(a_ext)
bc.apply(matC)
#------------------------------------------------------------------------------#
# correct (0,1)-block using dof mapping
#------------------------------------------------------------------------------#
# create dof mapping
mapping = restriction_map(extH1, intH1)
# correct column indices of B
import numpy as np
auxB = dlfn.as_backend_type(matB).mat()
rowptr, colind, values = auxB.getValuesCSR()
assert rowptr.size == m + 1
filter_fun = lambda x : mapping.has_key(x)
tol = 1e-12
for i in xrange(m):
    # get columns of this row
    del_ind = []
    cols = colind[rowptr[i]:rowptr[i+1]]
    for j, c in zip(range(rowptr[i], rowptr[i+1]), cols):
        if filter_fun(c):
            # correct dof index
            colind[j] = mapping[c]
        else:
            # mark for deletion
            del_ind.append(j)
    # checkout magnitude of values to delete
    check_mask = np.abs(values[del_ind]) < tol
    assert np.all(check_mask),\
                  """
                  Trying to delete a value with magnitude: {0:3.2e}
                  in row: {1} and column {2}""".format(
                          values[del_ind[np.where(np.logical_not(check_mask))[0][0]]],
                          i, colind[del_ind[np.where(np.logical_not(check_mask))[0][0]]])
    # delete indices from column and value arrays
    colind = np.delete(colind, del_ind)
    values = np.delete(values, del_ind)
    # correct row pointers
    cnt = len(del_ind)
    rowptr[i+1:] -= cnt
assert colind.max() < n
# scipy sparse matrices of B
from scipy.sparse import csr_matrix
spB = csr_matrix((values, colind, rowptr), shape=(m, n))
#------------------------------------------------------------------------------#
# create scipy block matrices
#------------------------------------------------------------------------------#
# scipy sparse matrices of A and C
spA = getScipySparseMatrix(matA)
spC = getScipySparseMatrix(matC)
spM = getScipySparseMatrix(matM)
# block system
from scipy.sparse import bmat
K = bmat(((spA, spB),
          (spB.transpose(), spC)))
M = bmat(((spM, None),
          (None, csr_matrix((n,n), dtype=np.float) )
          ))
system_matrix = M + K
#------------------------------------------------------------------------------#
# initial conditions
#------------------------------------------------------------------------------#
class InitialVectorPotential(dlfn.Expression):
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
initial_vector_pot = InitialVectorPotential(interior_id = intId, exterior_id = extId,
                                     cell_data = cellMeshFun, degree = 2)
tmp = dlfn.project(initial_vector_pot, intHCurl)
sol_A0.assign(tmp)
#------------------------------------------------------------------------------#
# output preparation
#------------------------------------------------------------------------------#
# DG spaces for projection
P1Grad = dlfn.VectorElement("DG", mesh.ufl_cell(), 0)
extH1Grad= dlfn.FunctionSpace(subMeshes[extId], P1Grad)
intH1Grad= dlfn.FunctionSpace(subMeshes[intId], P1Grad)
# output files
pvd_A = dlfn.File("./pvd/solution-A.pvd")
pvd_curlA = dlfn.File("./pvd/solution-curlA.pvd")
pvd_phi = dlfn.File("./pvd/solution-phi.pvd")
pvd_gradphi = dlfn.File("./pvd/solution-gradphi.pvd")
#------------------------------------------------------------------------------#
# time loop
#------------------------------------------------------------------------------#
step = 0
time = 0.0
# write initial condition
dlfn.File("initial-A.pvd") << (sol_A0, time)
# preconditioner
def jacobi_preconditioning(v):
    return v / system_matrix.diagonal()
from scipy.sparse.linalg import LinearOperator
P = LinearOperator(shape=system_matrix.shape,
                   dtype=system_matrix.dtype,
                   matvec=jacobi_preconditioning)
# allocation                   
rms_values = np.zeros((n_steps + 1, 4))
while time < t_end and step < n_steps:
    print "Iteration: {:08d}, ".format(step), "time = {0:10.5f},".format(time),\
            " time step = {0:5.4e}".format(dt)
    # assemble right-hand side
    rhs_int = dlfn.assemble(l_int)
    # scipy right-hand side vector
    b = np.hstack((rhs_int, np.zeros((n,))))
    # solve linear system
    from scipy.sparse.linalg import gmres
    x, info = gmres(system_matrix, b,
                    tol=1e-9 * np.linalg.norm(b),
                    maxiter=100,
                    M=P)
    assert info == 0
    x_int = x[:m]
    x_ext = x[m:]
    # assign solutions
    sol_A.vector()[:] = x_int
    sol_phi.vector()[:] = x_ext
    # time update
    time += dt
    step += 1
    # gradient projection
    grad_phi = dlfn.project(grad(sol_phi), extH1Grad)
    # curl projection
    curl_A = dlfn.project(curl(sol_A), intH1Grad)
    # write output
    pvd_A << (sol_A, time)
    pvd_curlA << (curl_A, time)
    pvd_phi << (sol_phi, time)
    pvd_gradphi << (grad_phi, time)
    # rms-values
    from math import sqrt
    rms_A = sqrt( dlfn.assemble(dot(sol_A, sol_A) * dV[intId]) / intV )
    rms_curlA = sqrt( dlfn.assemble(dot(curl_A, curl_A) * dV[intId]) / intV)
    rms_phi = sqrt( dlfn.assemble(sol_phi * sol_phi * dV[extId]) / extV)
    rms_gradphi = sqrt( dlfn.assemble(dot(grad_phi, grad_phi) * dV[extId]) / extV)
    print "rms_values (A, phi, curl(A), grad(phi)): " +\
        "{0:3.2e}, {1:3.2e}, {2:3.2e}, {3:3.2e}".format(rms_A, rms_phi, rms_curlA, rms_gradphi)
    rms_values[step,:] = np.array([rms_A, rms_phi, rms_curlA, rms_gradphi])
    # update solutions for next iteration
    sol_A0.assign(sol_A)
# plotting
rms_values = rms_values[:step-1,:]
import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=4, sharex=True)
for i in range(4):
    ax[i].plot(rms_values[:,i])
plt.savefig("rms-values.pdf")