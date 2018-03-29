from fenics import *

from afqsfenicsutil import write_vtk_f
import afqsrungekutta as rk

import ufl
ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True
parameters["form_compiler"]["quadrature_degree"]=3

mesh = ASDFFDSA

elemS1 = FiniteElement('CG', triangle, 1)
elemV1 = VectorElement('CG', triangle, 1)
element = MixedElement([ elemV1, elemS1 ])
M = FunctionSpace(mesh, element)

w = Function(M)
tw = TrialFunction(M)
Dw = TrialFunction(M)

A,phi = blahblahblah
