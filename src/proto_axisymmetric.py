from fenics import *

from afqsfenicsutil import write_vtk_f
import afqsrungekutta as rk

import ufl
ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True
parameters["form_compiler"]["quadrature_degree"]=3

mesh = Mesh("../cylinderInRectangle.xml")
cellids = MeshFunctionSizet(mesh,"../cylinderInRectangle_physical_region.xml")

dx = dx(subdomain_data=cellids)

S1 = FunctionSpace(mesh,"CG",1)
V1 = VectorFunctionSpace(mesh,"DG",0)
a = Function(S1)
ta = TestFunction(S1)
Da = TrialFunction(S1)

vz = 1.0

f_M = inner(ta,Da)*dx
f_R = -inner(grad(ta),grad(a))*dx - inner(ta,vz*a.dx(1))*dx(1)
f_K = derivative(f_R,a,Da)

bcs = []

from afqsrungekutta.rkfenics import RK_field_fenics
rkf = RK_field_fenics(1, [ a ], f_M, f_R, [f_K], bcs )

a.interpolate( Expression( "x[0]", degree=1 ) )
Tnow = 0.0
Tfinal = 100.0
DeltaT = Tfinal/1000.0
delta_outp = 0.1*DeltaT

step = rk.imRK.DIRK(DeltaT, rk.imRK.LDIRK['LSDIRK3'], [rkf] )


onum = 0
def output():
    global onum
    write_vtk_f("outs/viz_{0}.vtk".format(onum),
                mesh,
                {"a":a},
                {"b":project(curl(a),V1)})
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

