# -*- coding: utf-8 -*-
import dolfin as dlfn
import numpy as np
class ExactMagneticField(dlfn.Expression):
    """
    """
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

        self._r = lambda x: np.linalg.norm(x)
        self._theta = lambda x: np.arccos(x[2] / self._r(x)) if self._r(x) > 1e-12 else np.pi/2.
        self._phi = lambda x: np.arctan2(x[1], x[0])

    def _er(self, z):
        """
        Computes spherical basis vector :math:`\\boldsymbol{e}_r`
        according to

            .. math:: \\boldsymbol{e}_r =
                \\sin(\\theta)(\\cos(\\varphi)\\boldsymbol{e}_x + 
                \\sin(\\varphi)\\boldsymbol{e}_y)+
                \\cos(\\theta)\\boldsymbol{e}_z

        Parameters
        -----------
        z : ``numpy.ndarray``
            array of :math:`r` :math:`\\theta` and :math:`\\varphi` coordinates

        Returns
        -----------
        etheta : ``numpy.ndarray``
            array of computed components in :math:`\\boldsymbol{e}_x`,
            :math:`\\boldsymbol{e}_y` and :math:`\\boldsymbol{e}_z` basis.
        """
        if z[0] == 0.:
            er = np.array([1., 0., 0.])
        else:
            theta = z[1]
            phi = z[2]
            er = np.array([np.cos(phi)*np.sin(theta),
                           np.sin(phi)*np.sin(theta),
                           np.cos(theta)])
        return er

    def _etheta(self, z):
        """
        Computes spherical basis vector :math:`\\boldsymbol{e}_\\theta`
        according to

            .. math:: \\boldsymbol{e}_\\theta =
                \\cos(\\theta)(\\cos(\\varphi)\\boldsymbol{e}_x + 
                \\sin(\\varphi)\\boldsymbol{e}_y)-
                \\sin(\\theta)\\boldsymbol{e}_z

        Parameters
        -----------
        z : ``numpy.ndarray``
            array of :math:`r` :math:`\\theta` and :math:`\\varphi` coordinates

        Returns
        -----------
        etheta : ``numpy.ndarray``
            array of computed components in :math:`\\boldsymbol{e}_x`,
            :math:`\\boldsymbol{e}_y` and :math:`\\boldsymbol{e}_z` basis.
        """
        if z[0] == 0.:
            etheta = np.array([0., 0., -1.])
        else:
            theta = z[1]
            phi = z[2]
            etheta = np.array([np.cos(phi)*np.cos(theta),
                               np.sin(phi)*np.cos(theta),
                               -np.sin(theta)])
        return etheta

    def _ephi(self, z):
        """
        Computes spherical basis vector :math:`\\boldsymbol{e}_\\varphi`
        according to

            .. math:: \\boldsymbol{e}_\\varphi =
                -\\sin(\\varphi)\\boldsymbol{e}_x +
                \\cos(\\varphi)\\boldsymbol{e}_y

        Parameters
        -----------
        z : ``numpy.ndarray``
            array of :math:`r` :math:`\\theta` and :math:`\\varphi` coordinates

        Returns
        -----------
        etheta : ``numpy.ndarray``
            array of computed components in :math:`\\boldsymbol{e}_x`,
            :math:`\\boldsymbol{e}_y` and :math:`\\boldsymbol{e}_z` basis.
        """
        if z[0] == 0.:
            ephi = np.array([0., 1., 0.])
        else:
            phi = z[2]
            ephi = np.array([-np.sin(phi), np.cos(phi), 0.])
        return ephi

    def value_shape(self):
        return (self._dim,)

    def eval_cell(self, value, x, ufl_cell):
        # assign value depending on domain
        if self._cell_data[ufl_cell.index] == self._interior_id:
            value[0] = 0.0
            value[1] = 0.0
            value[2] = 2./3.
        elif self._cell_data[ufl_cell.index] == self._exterior_id:
            # spherical coordinates and basis vectors
            z = np.array([self._r(x), self._theta(x), self._phi(x)])
            er = self._er(z)
            etheta = self._etheta(z)
            for d in xrange(self._dim):
                value[d] = (2. * np.cos(z[1])  * er[d] \
                             + np.sin(z[1]) * etheta[d]) / (3. * z[0]**3)
        else:
            raise ValueError()

class ExactScalarPotential(dlfn.Expression):
    """
    """
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

        self._r = lambda x: np.linalg.norm(x)
        self._theta = lambda x: np.arccos(x[2] / self._r(x)) if self._r(x) > 1e-12 else np.pi/2.
        self._phi = lambda x: np.arctan2(x[1], x[0])

    def eval_cell(self, value, x, ufl_cell):
        # assign value depending on domain
        if self._cell_data[ufl_cell.index] == self._exterior_id:
            # spherical coordinates and basis vectors
            r, theta = self._r(x), self._theta(x)
            value[0] = np.cos(theta) / (3. * r**2)
        else:
            raise ValueError()

class ExactVectorPotential(dlfn.Expression):
    """
    """
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

        self._r = lambda x: np.linalg.norm(x)
        self._theta = lambda x: np.arccos(x[2] / self._r(x)) if self._r(x) > 1e-12 else np.pi/2.
        self._phi = lambda x: np.arctan2(x[1], x[0])

    def _er(self, z):
        """
        Computes spherical basis vector :math:`\\boldsymbol{e}_r`
        according to

            .. math:: \\boldsymbol{e}_r =
                \\sin(\\theta)(\\cos(\\varphi)\\boldsymbol{e}_x +
                \\sin(\\varphi)\\boldsymbol{e}_y)+
                \\cos(\\theta)\\boldsymbol{e}_z

        Parameters
        -----------
        z : ``numpy.ndarray``
            array of :math:`r` :math:`\\theta` and :math:`\\varphi` coordinates

        Returns
        -----------
        etheta : ``numpy.ndarray``
            array of computed components in :math:`\\boldsymbol{e}_x`,
            :math:`\\boldsymbol{e}_y` and :math:`\\boldsymbol{e}_z` basis.
        """
        if z[0] == 0.:
            er = np.array([1., 0., 0.])
        else:
            theta = z[1]
            phi = z[2]
            er = np.array([np.cos(phi)*np.sin(theta),
                           np.sin(phi)*np.sin(theta),
                           np.cos(theta)])
        return er

    def _etheta(self, z):
        """
        Computes spherical basis vector :math:`\\boldsymbol{e}_\\theta`
        according to

            .. math:: \\boldsymbol{e}_\\theta =
                \\cos(\\theta)(\\cos(\\varphi)\\boldsymbol{e}_x +
                \\sin(\\varphi)\\boldsymbol{e}_y)-
                \\sin(\\theta)\\boldsymbol{e}_z

        Parameters
        -----------
        z : ``numpy.ndarray``
            array of :math:`r` :math:`\\theta` and :math:`\\varphi` coordinates

        Returns
        -----------
        etheta : ``numpy.ndarray``
            array of computed components in :math:`\\boldsymbol{e}_x`,
            :math:`\\boldsymbol{e}_y` and :math:`\\boldsymbol{e}_z` basis.
        """
        if z[0] == 0.:
            etheta = np.array([0., 0., -1.])
        else:
            theta = z[1]
            phi = z[2]
            etheta = np.array([np.cos(phi)*np.cos(theta),
                               np.sin(phi)*np.cos(theta),
                               -np.sin(theta)])
        return etheta

    def _ephi(self, z):
        """
        Computes spherical basis vector :math:`\\boldsymbol{e}_\\varphi`
        according to

            .. math:: \\boldsymbol{e}_\\varphi =
                -\\sin(\\varphi)\\boldsymbol{e}_x +
                \\cos(\\varphi)\\boldsymbol{e}_y

        Parameters
        -----------
        z : ``numpy.ndarray``
            array of :math:`r` :math:`\\theta` and :math:`\\varphi` coordinates

        Returns
        -----------
        etheta : ``numpy.ndarray``
            array of computed components in :math:`\\boldsymbol{e}_x`,
            :math:`\\boldsymbol{e}_y` and :math:`\\boldsymbol{e}_z` basis.
        """
        if z[0] == 0.:
            ephi = np.array([0., 1., 0.])
        else:
            phi = z[2]
            ephi = np.array([-np.sin(phi), np.cos(phi), 0.])
        return ephi

    def value_shape(self):
        return (self._dim,)

    def eval_cell(self, value, x, ufl_cell):
        # assign value depending on domain
        if self._cell_data[ufl_cell.index] == self._interior_id:
            # spherical coordinates and basis vectors
            z = np.array([self._r(x), self._theta(x), self._phi(x)])
            ephi = self._etheta(z)
            for d in xrange(self._dim):
                value[d] =  (z[0] * np.sin(z[1])  * ephi[d]) / 3.
        else:
            raise ValueError()