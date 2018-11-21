# -*- coding: utf-8 -*-
import dolfin as dlfn
import numpy as np
class ExactMagneticField(dlfn.Expression):
    def __init__(self, **kwargs):
        # user input check
        assert isinstance(kwargs["cell_data"], dlfn.MeshFunctionSizet)
        self._cell_data = kwargs["cell_data"]
        assert isinstance(kwargs["interior_id"], int)
        assert isinstance(kwargs["exterior_id"], int)
        self._interior_id = kwargs["interior_id"]
        self._exterior_id = kwargs["exterior_id"]
        dim = self._cell_data.mesh().topology().dim()
        if dim != 3:
            raise NotImplementedError()

    def value_shape(self):
        return (3,)

    def eval_cell(self, value, x, ufl_cell):
        r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
        # assign value depending on domain
        if self._cell_data[ufl_cell.index] == self._interior_id:
            value[0] = 0.
            value[1] = 0.
            value[2] = -1./3.
        elif self._cell_data[ufl_cell.index] == self._exterior_id:
            value[0] = x[0] * x[2] / (3. * r**(5./2.))
            value[1] = x[1] * x[2] / (3. * r**(5./2.))
            value[2] = (2. * x[2]**2 - x[0]**2 - x[1]**2) / (3. * r**(5./2.))
        else:
            raise ValueError()

class ExactScalarPotential(dlfn.Expression):
    def __init__(self, **kwargs):
        # user input check
        assert isinstance(kwargs["cell_data"], dlfn.MeshFunctionSizet)
        self._cell_data = kwargs["cell_data"]
        assert isinstance(kwargs["interior_id"], int)
        assert isinstance(kwargs["exterior_id"], int)
        self._interior_id = kwargs["interior_id"]
        self._exterior_id = kwargs["exterior_id"]
        dim = self._cell_data.mesh().topology().dim()
        if dim != 3:
            raise NotImplementedError()

    def eval_cell(self, value, x, ufl_cell):
        r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
        # assign value depending on domain
        if self._cell_data[ufl_cell.index] == self._interior_id:
            value[0] = x[2] / 3.
        elif self._cell_data[ufl_cell.index] == self._exterior_id:
            value[0] = x[2] / 3. / r**(3./2.)
        else:
            raise ValueError()

class ExactVectorPotential(dlfn.Expression):
    def __init__(self, **kwargs):
        # user input check
        assert isinstance(kwargs["cell_data"], dlfn.MeshFunctionSizet)
        self._cell_data = kwargs["cell_data"]
        assert isinstance(kwargs["interior_id"], int)
        assert isinstance(kwargs["exterior_id"], int)
        self._interior_id = kwargs["interior_id"]
        self._exterior_id = kwargs["exterior_id"]
        dim = self._cell_data.mesh().topology().dim()
        if dim != 3:
            raise NotImplementedError()

    def value_shape(self):
        return (3,)

    def eval_cell(self, value, x, ufl_cell):
        # assign value depending on domain
        r = np.sqrt(x[0]**2 +   x[1]**2 +   x[2]**2)
        if self._cell_data[ufl_cell.index] == self._interior_id:
            value[0] = - x[1] / 6.
            value[1] = x[0] / 6.
            value[2] = 0.
        elif self._cell_data[ufl_cell.index] == self._exterior_id:
            value[0] = - x[1] / (3. * r**3)
            value[1] = x[0] / (3. * r**3)
            value[2] = 0.
        else:
            raise ValueError()