import dolfin as dlfn
from afqsfenicsutil.my_restriction_map  import *

def transferCellMeshFun(fun, subfun, zero):
    parent_cell_indices = subfun.mesh().data().array("parent_cell_indices", 3).astype(np.int)
    for cell_id, parent_cell_id in enumerate(parent_cell_indices):
            if fun[parent_cell_id] != zero:
                subfun.set_value(cell_id, fun[parent_cell_id])

def transferFacetMeshFun(fun, subfun, zero):
    # get meshes
    mesh = fun.mesh()
    submesh = subfun.mesh()
    # cell mapping
    parent_cell_indices = submesh.data().array("parent_cell_indices", 3).astype(np.int)
    parent_vert_indices = submesh.data().array("parent_vertex_indices", 0).astype(np.int)
    # space dimension
    dim = fun.mesh().geometry().dim()
    # initialize facets    
    submesh.init(dim - 1, dim)
    # loop over cell on parent mesh
    for parent_cell in dlfn.cells(mesh):
        parent_cell_id = parent_cell.index()
        # loop over facet of parent cell
        for parent_facet_id in parent_cell.entities(dim - 1).astype(np.int):
            if fun[parent_facet_id] != zero:
                # get parent vertices
                parent_facet = dlfn.Facet(mesh, parent_facet_id)
                parent_facet_vert_ids = parent_facet.entities(0)
                # find matching cell in sub mesh (invert mapping)
                cell_id = np.where(parent_cell_indices == parent_cell_id)[0]
                if cell_id.size > 0:
                    assert cell_id.size == 1
                    cell = dlfn.Cell(submesh, cell_id[0])
                    # loop over child facet
                    for facet_id in cell.entities(dim - 1):
                        facet = dlfn.Facet(submesh, facet_id)
                        facet_match = True
                        for vert_id in facet.entities(0):
                            if parent_vert_indices[vert_id] not in parent_facet_vert_ids:
                                facet_match = False
                                break
                        if facet_match == True:
                            subfun.set_value(facet_id, fun[parent_facet_id])
                            break

def transferVertexMeshFun(fun, subfun, zero):
    parent_vert_indices = subfun.mesh().data().array("parent_vertex_indices", 0).astype(np.int)
    # loop over vertices
    for vert_id, parent_vert_id in enumerate(parent_vert_indices):
        if fun[parent_vert_id] != zero:
            subfun.set_value(vert_id, fun[parent_vert_id])

def transferMeshFunToSubMesh(subMesh, meshFun):
    """
    This method transfers a mesh function to a submesh.
    """
    # input check
    meshFunTypes = (dlfn.MeshFunctionSizet, dlfn.MeshFunctionBool,
                    dlfn.MeshFunctionDouble, dlfn.MeshFunctionInt)
    zeroType = (0, False, 0.0, 0)
    assert isinstance(subMesh, dlfn.SubMesh)
    assert isinstance(meshFun, meshFunTypes)
    parentMesh = meshFun.mesh()
    assert isinstance(parentMesh, dlfn.Mesh)
    dim = parentMesh.geometry().dim()
    assert subMesh.geometry().dim() == dim
    # auxiliary variables
    dimMeshFun = meshFun.dim()
    typeIndicator = [isinstance(meshFun, x) for x in meshFunTypes]
    typeIndex = typeIndicator.index(True)
    # initialize facet functions on submesh
    subMeshFun = meshFunTypes[typeIndex](subMesh, dimMeshFun)
    subMeshFun.set_all(zeroType[typeIndex])
    # parent information    
    # loop over cells, facets, edges or vertices
    if dimMeshFun == dim:
        # cell mesh function
        transferCellMeshFun(meshFun, subMeshFun, zeroType[typeIndex])
    elif dimMeshFun == dim - 1:
        # facet mesh function
        transferFacetMeshFun(meshFun, subMeshFun, zeroType[typeIndex])
    elif dimMeshFun == dim - 2 and dim != 2:
        # edge mesh function
        raise NotImplementedError()
    elif dimMeshFun == 0:
        # vertex mesh function
        transferVertexMeshFun(meshFun, subMeshFun, zeroType[typeIndex])
    else:
        raise ValueError()
    return subMeshFun

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
for i in subIds:
    facetSubMeshFuns[i] = transferMeshFunToSubMesh(subMeshes[i], facetMeshFun)
    pvd_file = dlfn.File("factfun-{}.pvd".format(i))
    pvd_file << facetSubMeshFuns[i]
#------------------------------------------------------------------------------#
# function spaces, test/trial functions, ...
#------------------------------------------------------------------------------#
# finite element spaces
P1Curl = dlfn.FiniteElement("N1curl", mesh.ufl_cell(), 2)
P1 = dlfn.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
# function spaces
intHCurl = dlfn.FunctionSpace(subMeshes[intId], P1Curl)
intH1 = dlfn.FunctionSpace(subMeshes[intId], P1)
extH1 = dlfn.FunctionSpace(subMeshes[extId], P1)
# size of function spaces
m = intHCurl.dim()
n = extH1.dim()
nn = intH1.dim()
# test and trial functions
A = dlfn.TrialFunction(intHCurl)
B = dlfn.TestFunction(intHCurl)
phi = dlfn.TrialFunction(extH1)
auxphi = dlfn.TrialFunction(intH1)
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
from dolfin import curl, inner, dot, cross
jumpM = dlfn.Constant((0.0, 0.0, -1.0))
a_int = inner(curl(A), curl(B)) * dV[intId]
l_int = dot(cross(normals[intId], jumpM), B) * dA[intId](intrfcId)
b_int = auxphi * dot(normals[intId], curl(B)) * dA[intId](intrfcId)
# assemble (0,0)-block
matA = dlfn.assemble(a_int)
rhs_int = dlfn.assemble(l_int)
# assemble (0,1)-block
matB = dlfn.assemble(b_int)
#------------------------------------------------------------------------------#
# weak forms and assembly in exterior
#------------------------------------------------------------------------------#
# linear forms in exterior domain
from dolfin import grad
a_ext = inner(grad(phi), grad(psi)) * dV[extId]
l_ext = dlfn.Constant(0.) * psi * dV[extId]
# apply boundary condition
bc = dlfn.DirichletBC(extH1, dlfn.Constant(0.),
                      facetSubMeshFuns[extId], bndryId)
# assemble (1,1)-block
matC, rhs_ext = dlfn.assemble_system(a_ext, l_ext, bcs=bc)
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
indptr, indices, data = dlfn.as_backend_type(matA).mat()\
                            .getValuesCSR()
spA = csr_matrix((data, indices, indptr), shape=(m, m))
indptr, indices, data = dlfn.as_backend_type(matC).mat()\
                            .getValuesCSR()
spC = csr_matrix((data, indices, indptr), shape=(n, n))
# block system
from scipy.sparse import bmat
M = bmat(((spA, spB), (spB.transpose(), spC)))
# block right-hand side
b = np.hstack((rhs_int, rhs_ext))
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
x_ext = x[m:]
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
# DG spaces for projection
P1Grad = dlfn.VectorElement("DG", mesh.ufl_cell(), 0)
extH1Grad= dlfn.FunctionSpace(subMeshes[extId], P1Grad)
intH1Grad= dlfn.FunctionSpace(subMeshes[intId], P1Grad)
# gradient projection
grad_sol_ext = dlfn.project(grad(sol_ext), extH1Grad)
# curl projection
curl_sol_int = dlfn.project(curl(sol_int), intH1Grad)
# write output
dlfn.File("solution-A.pvd") << sol_int
dlfn.File("solution-curlA.pvd") << curl_sol_int
dlfn.File("solution-phi.pvd") << sol_ext
dlfn.File("solution-gradphi.pvd") << grad_sol_ext
#------------------------------------------------------------------------------#
# evaluate on line
#------------------------------------------------------------------------------#
n = 50
ri, ro = 1.0 , 3.0
r = np.linspace(ri, ro, num= n + 1)[1:]
theta = 0. * np.pi
x, z = r * np.cos(theta), r * np.sin(theta)
plt_val = np.zeros_like(x)
for i in xrange(plt_val.size):
    plt_val[i] = sol_ext([x[i], 0., z[i]])
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(r, plt_val, 'x--')
plt_val = np.zeros((x.size, 2))
for i in xrange(n):
    xx = np.array([x[i], 0., z[i]])
    er = xx / np.linalg.norm(xx)
    etheta = np.array([xx[0] * xx[2], xx[1] * xx[2], -xx[0]**2-xx[2]**2])
    etheta /= np.linalg.norm(xx) * np.sqrt(xx[0]**2+xx[1]**2)
    plt_val[i,0] = np.dot(grad_sol_ext([x[i], 0., z[i]]), er)
    plt_val[i,1] = np.dot(grad_sol_ext([x[i], 0., z[i]]), etheta)
ax[1].plot(r, plt_val[:,0], 'x--', label="radial")
ax[1].plot(r, plt_val[:,1], 'x--', label="tangential")
ax[1].legend()
plt.savefig("ext-sol-radial.png")
# evaluate
n = 50
ri, ro = 1.0 , 3.0
r = (ri + ro) / 2.
theta = np.linspace(0., np.pi, num = n + 2)[1:-1]
x, z = r * np.cos(theta), r * np.sin(theta)
plt_val = np.zeros_like(x)
for i in xrange(plt_val.size):
    plt_val[i] = sol_ext([x[i], 0., z[i]])
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(theta, plt_val, 'x--')
plt_val = np.zeros((x.size, 2))
for i in xrange(n):
    xx = np.array([x[i], 0., z[i]])
    er = xx / np.linalg.norm(xx)
    etheta = np.array([xx[0] * xx[2], xx[1] * xx[2], -xx[0]**2-xx[2]**2])
    etheta /= np.linalg.norm(xx) * np.sqrt(xx[0]**2+xx[1]**2)
    plt_val[i,0] = np.dot(grad_sol_ext([x[i], 0., z[i]]), er)
    plt_val[i,1] = np.dot(grad_sol_ext([x[i], 0., z[i]]), etheta)
ax[1].plot(theta, plt_val[:,0], 'x--', label="radial")
ax[1].plot(theta, plt_val[:,1], 'x--', label="tangential")
plt.savefig("ext-sol-theta.png")