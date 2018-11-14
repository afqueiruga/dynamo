import dolfin as dlfn
from afqsfenicsutil.my_restriction_map  import *
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
# import initial mesh
mesh_name = "../../meshes/sphereInCube"
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
for i in subIds:
    facetSubMeshFuns[i] = transferMeshFunToSubMesh(subMeshes[i], facetMeshFun)
    pvd_file = dlfn.File("factfun-{}.pvd".format(i))
    pvd_file << facetSubMeshFuns[i]
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
n = intH1.dim()
o = extH1.dim()
# test and trial functions
A = dlfn.TrialFunction(intHCurl)
B = dlfn.TestFunction(intHCurl)
u = dlfn.TrialFunction(intH1)
v = dlfn.TestFunction(intH1)
phi = dlfn.TrialFunction(extH1)
psi = dlfn.TestFunction(extH1)
# measures and normal vectors
dA, dV, normals = dict(), dict(), dict()
for i in subIds:
    dA[i] = dlfn.Measure("ds", subMeshes[i], subdomain_data=facetSubMeshFuns[i])
    dV[i] = dlfn.Measure("dx", subMeshes[i])
    normals[i] = dlfn.FacetNormal(subMeshes[i])
#------------------------------------------------------------------------------#
# weak forms and assembly in interior
#------------------------------------------------------------------------------#
# linear forms in interior domain
from dolfin import curl, inner, dot, cross, grad
jumpM = dlfn.Constant((0.0, 0.0, -1.0))
a_int = inner(curl(A), curl(B)) * dV[intId] 
b_int = dot(A, grad(v)) * dV[intId]
c_int = u * dot(normals[intId], curl(B)) * dA[intId](intrfcId)
l_int = dot(cross(normals[intId], jumpM), B) * dA[intId](intrfcId)
# assemble (0,0)-block
matA = dlfn.assemble(a_int)
rhs_int = dlfn.assemble(l_int)
# assemble (0,1)-block
matB = dlfn.assemble(b_int)
# assemble (0,2)-block
matC = dlfn.assemble(c_int)
#------------------------------------------------------------------------------#
# weak forms and assembly in exterior
#------------------------------------------------------------------------------#
# linear forms in exterior domain
a_ext = inner(grad(phi), grad(psi)) * dV[extId]
l_ext = dlfn.Constant(0.) * psi * dV[extId]
# apply boundary condition
bc = dlfn.DirichletBC(extH1, dlfn.Constant(0.),
                      facetSubMeshFuns[extId], bndryId)
# assemble (2,2)-block
matD, rhs_ext = dlfn.assemble_system(a_ext, l_ext, bcs=bc)
#------------------------------------------------------------------------------#
# correct (0,1)-block using dof mapping
#------------------------------------------------------------------------------#
# create dof mapping
mapping = restriction_map(extH1, intH1)
# correct column indices of B
import numpy as np
auxC = dlfn.as_backend_type(matC).mat()
rowptr, colind, values = auxC.getValuesCSR()
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
assert colind.max() < o
# scipy sparse matrices of B
from scipy.sparse import csr_matrix
spC = csr_matrix((values, colind, rowptr), shape=(m, o))
#------------------------------------------------------------------------------#
# create scipy block matrices
#------------------------------------------------------------------------------#
# scipy sparse matrices of A and C
spA = getScipySparseMatrix(matA)
spB = getScipySparseMatrix(matB)
spD = getScipySparseMatrix(matD)
# block system
from scipy.sparse import bmat
M = bmat(((spA, spB.transpose(), spC),
          (spB, None, None), 
          (spC.transpose(), None, spD)))
# block right-hand side
b = np.hstack((rhs_int, np.zeros((n,)), rhs_ext))
# plot sparsity pattern
import matplotlib.pyplot as plt
plt.figure()
plt.spy(M)
plt.savefig("sparsity_pattern.png")
#------------------------------------------------------------------------------#
# linear solve
#------------------------------------------------------------------------------#
# solve linear system
from scipy.sparse.linalg import spsolve
x = spsolve(M, b)
x_int = x[:m]
x_ext = x[m+n:]
#------------------------------------------------------------------------------#
# finite element solutions
#------------------------------------------------------------------------------#
sol_int = dlfn.Function(intHCurl)
sol_int.vector()[:] = x_int
sol_ext = dlfn.Function(extH1)
sol_ext.vector()[:] = x_ext
#------------------------------------------------------------------------------#
# output element solutions
#------------------------------------------------------------------------------#
sol_A = sol_int
# DG spaces for projection
P1Grad = dlfn.VectorElement("DG", mesh.ufl_cell(), 0)
extH1Grad= dlfn.FunctionSpace(subMeshes[extId], P1Grad)
intH1Grad= dlfn.FunctionSpace(subMeshes[intId], P1Grad)
# gradient projection
grad_sol_ext = dlfn.project(grad(sol_ext), extH1Grad)
# curl projection
curl_sol_int = dlfn.project(curl(sol_A), intH1Grad)
# write output
dlfn.File("solution-A.pvd") << sol_int
dlfn.File("solution-curlA.pvd") << curl_sol_int
dlfn.File("solution-phi.pvd") << sol_ext
dlfn.File("solution-gradphi.pvd") << grad_sol_ext