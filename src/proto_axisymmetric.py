from fenics import *

from afqsfenicsutil import write_vtk_f
import afqsrungekutta as rk

import ufl
ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True
parameters["form_compiler"]["quadrature_degree"]=3

mesh = Mesh("../cylinderInRectangle.xml")
cellids = MeshFunctionSizet(mesh,"../cylinderInRectangle_physical_region.xml")

boty = mesh.coordinates().min(0)[1]
topy = mesh.coordinates().max(0)[1]
# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(x[1] - boty <  DOLFIN_EPS and
                    x[1] - boty > -DOLFIN_EPS and on_boundary)
    def map(self, x, y):
        y[0] = x[0]
        y[1] = x[1] + (topy-boty)

# Create periodic boundary condition
pbc = PeriodicBoundary()

dx = dx(subdomain_data=cellids)
x = SpatialCoordinate(mesh)
S1 = FunctionSpace(mesh,"CG",1, constrained_domain=pbc)
V1 = VectorFunctionSpace(mesh,"DG",0)
a = Function(S1)
ta = TestFunction(S1)
Da = TrialFunction(S1)

B = as_tensor( (-a.dx(1), a.dx(0) + 1/x[0]*a) )
tB = as_tensor( (-ta.dx(1), ta.dx(0) + 1/x[0]*ta) )
# tB = curl(ta) + 1/x[0]*ta*Constant((0,1))
J = Function(S1)

vz = 1000.0
Jtheta = 1.0
# - inner(ta,vz*a.dx(1))*dx(1)
f_M = inner(ta,Da)*x[0]*dx
# f_R = -inner(grad(ta),grad(a))*x[0]*dx - ta*grad(a)[0]*dx - inner(ta,Jtheta)*x[0]*dx(3)
f_R = -inner(tB,B)*x[0]*dx - inner(ta,vz*a.dx(0))*dx(1) - inner(ta,Jtheta)*x[0]*dx(3)
f_K = derivative(f_R,a,Da)

bcs = [
    DirichletBC(S1,0.0,
                CompiledSubDomain("x[0]>L && on_boundary",L=mesh.coordinates().max(0)[1]-1.0e-6))
]

from afqsrungekutta.rkfenics import RK_field_fenics
rkf = RK_field_fenics(1, [ a ], f_M, f_R, [f_K], bcs )
rkf.maxnewt = 1

# a.interpolate( Expression( "x[0]", degree=1 ) )
Tnow = 0.0
Tfinal = 10000.0
DeltaT = Tfinal/1000.0
delta_outp = 0.1*DeltaT

step = rk.imRK.DIRK(DeltaT, rk.imRK.LDIRK['LSDIRK3'], [rkf] )

onum = 0
def output():
    global onum
    write_vtk_f("outs/viz_{0}.vtk".format(onum),
                mesh,
                {"a":a},
                {"b":project(B,V1)})
    onum+=1

output()
next_outp = delta_outp-1.0e-1
while Tnow < Tfinal:
    step.march()
    if Tnow >= next_outp:
        output()
        next_outp += delta_outp
        print "Wrote at ", Tnow
    Tnow += DeltaT

