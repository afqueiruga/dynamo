from fenics import *

from afqsfenicsutil import write_vtk_f
import afqsrungekutta as rk

# element formulation
c = mesh.ufl_cell()
elemV = VectorElement("CG", c, p_deg)
elemP = FiniteElement("CG", c, p_deg - 1)
Uh = FunctionSpace(mesh, elemV)
Wh = FunctionSpace(mesh, elemV * elemP)
# non-dimensional parameters
Re = Constant(1200.0)
Fr = Constant(1.0)
# volumetric force
f = Constant((0.,0.,-9.81))
# surface and volume element
dA = Measure("ds", domain = mesh)
dV = Measure("dx", domain = mesh)
# solution functions for saddle point problem
sol = Function(Wh)
u, p = split(sol)
# creating trial functions
(v, q) = TestFunctions(Wh)
# components of weak form
def a(phi, psi):
    return inner(grad(phi), grad(psi))
def b(phi, psi):
    return div(phi) * psi
def c(phi, chi, psi):
    return dot(dot(grad(chi), phi), psi)
# weak form
F = (1.0/Re * a(u, v) - b(u, q) - b(v, p) + c(u, u, w) 
	- 1.0/Fr**2 * dot(f, v) ) * dV
