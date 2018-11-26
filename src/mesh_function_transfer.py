# -*- coding: utf-8 -*-
import dolfin as dlfn
import numpy as np
def _transferCellMeshFun(fun, subfun, zero):
    parent_cell_indices = subfun.mesh().data().array("parent_cell_indices", 3).astype(np.int)
    for cell_id, parent_cell_id in enumerate(parent_cell_indices):
            if fun[parent_cell_id] != zero:
                subfun.set_value(cell_id, fun[parent_cell_id])
def _transferFacetMeshFun(fun, subfun, zero):
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

def _transferVertexMeshFun(fun, subfun, zero):
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
        _transferCellMeshFun(meshFun, subMeshFun, zeroType[typeIndex])
    elif dimMeshFun == dim - 1:
        # facet mesh function
        _transferFacetMeshFun(meshFun, subMeshFun, zeroType[typeIndex])
    elif dimMeshFun == dim - 2 and dim != 2:
        # edge mesh function
        raise NotImplementedError()
    elif dimMeshFun == 0:
        # vertex mesh function
        _transferVertexMeshFun(meshFun, subMeshFun, zeroType[typeIndex])
    else:
        raise ValueError()
    return subMeshFun