#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import dolfin as dlfn
from exact_solutions import ExactMagneticField
# physical ids
"""
surfaces:
    1   interior spherical surface
    2   exterior surface
volumes:
    1   interior material volume
    2   exterior vacuous volume
"""
intVolId = 1
extVolId = 2
intrfcId = 1
bndryId = 2
# import initial mesh
mesh_name = "../../meshes/sphereInCube"
mesh = dlfn.Mesh(mesh_name + ".xml" )
dim = mesh.geometry().dim()
# facet and cell id markers
cellIds = dlfn.MeshFunctionSizet(mesh, mesh_name + "_physical_region.xml")
facetIds = dlfn.MeshFunctionSizet(mesh, mesh_name + "_facet_region.xml")
# measures
dA = dlfn.Measure("dS", mesh, subdomain_data=facetIds)
dGamma = dlfn.Measure("ds", mesh, subdomain_data=facetIds)
dV = dlfn.Measure("dx", mesh, subdomain_data=cellIds)
# function spaces
P1Curl = dlfn.FiniteElement("N1curl", mesh.ufl_cell(), 1)
P1 = dlfn.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
Vh = dlfn.FunctionSpace(mesh, P1Curl * P1)
# rhs function
jumpM = dlfn.Constant((0.0, 0.0, -1.0))
# trial functions
A, phi = dlfn.TrialFunctions(Vh)
# test functions
delA, psi = dlfn.TestFunctions(Vh)
# geometric objects
n = dlfn.FacetNormal(mesh)
P = dlfn.Identity(dim) - dlfn.outer(n,n)
# Dirichlet boundary condition on exterior surface
bcA = dlfn.DirichletBC(Vh.sub(0), dlfn.Constant((0.0, 0.0, 0.0)), facetIds, bndryId)
bcPhi = dlfn.DirichletBC(Vh.sub(1), dlfn.Constant(0.0), facetIds, bndryId)
# bilinear form 
a = dlfn.dot(dlfn.curl(A), dlfn.curl(delA)) * dV() \
    + dlfn.dot(A, dlfn.grad(psi)) * dV() \
    + dlfn.dot(dlfn.grad(phi), delA) * dV()
# rhs form
l = dlfn.dot(dlfn.cross(n("+"), jumpM("+")), delA("+")) * dA(intrfcId)
# compute solution
sol = dlfn.Function(Vh)
lin_problem = dlfn.LinearVariationalProblem(a, l, sol, bcs=[bcA, bcPhi])
lin_solver = dlfn.LinearVariationalSolver(lin_problem)
lin_solver_parameters = lin_solver.parameters
lin_solver.solve()
# sub solutions
solA = sol.sub(0)
solPhi = sol.sub(1)
# output to pvd
pvd_A = dlfn.File("solution-A.pvd")
pvd_A << solA
# solving for magnetic field
Wh = dlfn.VectorFunctionSpace(mesh, "CG", 1)
curlA = dlfn.project(dlfn.curl(solA), Wh)
# output to pvd
pvd_H = dlfn.File("solution-curlA.pvd")
pvd_H << curlA
# compute error
exact_field = ExactMagneticField(element = Wh.ufl_element(),
                                 cell_data = cellIds,
                                 interior_id = intVolId,
                                 exterior_id = extVolId)
exactH = dlfn.project(exact_field, Wh)
l2_error = dlfn.errornorm(curlA, exactH, degree_rise=0)
h1_error = dlfn.errornorm(curlA, exactH, degree_rise=0, norm_type="H1")
h1curl_error = dlfn.errornorm(curlA, exactH, degree_rise=0, norm_type="Hcurl")
print h1_error, l2_error, h1curl_error