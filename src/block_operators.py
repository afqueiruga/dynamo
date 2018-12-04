# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 12:09:49 2018

@author: sg
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator

def check_type(name,default='float64'):
    """ Try to convert input into a numpy.dtype object """
    value = str(name) if name is not None else default
    if value not in ['float32', 'float64', 'complex64', 'complex128']:
        raise ValueError("Incorrect type (%s)" % value)
    return np.dtype(value)


def combined_type(dtype1,dtype2):
    """ Return a type that is compatible with dtype1 and dtype2 """
    d1 = check_type(dtype1)
    d2 = check_type(dtype2)
    return (d1.type(1)*d2.type(1)).dtype

class BlockLinearOperator(LinearOperator):
    """
    Documentation is missing
    
    TODO: Fix this!
    """
    def __init__(self, m, n):
        self._sp_operator = None
        self._sp_format = None
        self._m = m
        self._n = n
        self._operators = np.empty((m, n), dtype=np.object)
        self._rows = np.zeros(m, dtype=int)
        self._cols = np.zeros(n, dtype=int)
    
    def __getitem__(self, key):
        return self._operators[key]

    def __setitem__(self, key, operator):
        import scipy.sparse as sp_sparse            
        assert isinstance(operator, (LinearOperator, sp_sparse.spmatrix) )
        if self._rows[key[0]] != 0:
            if operator.shape[0] != self._rows[key[0]]:
                raise ValueError("Incompatible number of rows")
        else:
            self._rows[key[0]] = operator.shape[0]

        if self._cols[key[1]] != 0:
            if operator.shape[1] != self._cols[key[1]]:
                raise ValueError("Incompatible number of columns")
        else:
            self._cols[key[1]] = operator.shape[1]
        self._operators[key] = operator

    def _fill_complete(self):
        if (0 in self._rows) or (0 in self._cols):
            return False
        return True

    def _matvec(self, x):
        if x.ndim == 1:
            x_new = np.expand_dims(x, 1)
            return self.matvec(x_new).ravel()
        if not self._fill_complete():
            raise ValueError("Not all rows or columns contain operators.")
        row_dim = 0
        res = np.zeros((self.shape[0], x.shape[1]), dtype = self.dtype)
        for i in range(self._m):
            col_dim = 0
            local_res = res[row_dim:row_dim + self._rows[i], :]
            for j in range(self._n):
                local_x = x[col_dim:col_dim + self._cols[j], :]
                if self._operators[i, j] is not None:
                    op_is_complex = np.iscomplexobj(self._operators[i, j].dtype.type(1))
                    if np.iscomplexobj(x) and not op_is_complex:
                        local_res[:] += (self._operators[i, j].dot(np.real(local_x)) +
                                         1j * self._operators[i, j].dot(np.imag(local_x)))
                    else:
                        local_res[:] += self._operators[i, j].dot(local_x)
                col_dim += self._cols[j]
            row_dim += self._rows[i]
        return res

    def _get_shape(self):
        return (np.sum(self._rows), np.sum(self._cols))

    def _get_dtype(self):
        d = 'float64'
        for obj in self._operators.ravel():
            if obj is not None:
                d = combined_type(d, obj.dtype)
        return d

    def _get_ndims(self):
        return (self._m, self._n)

    def _get_row_dimensions(self):
        return self._rows

    def _get_column_dimensions(self):
        return self._cols

    def _set_sparse_format(self, sp_format):
        assert self._sp_format is None
        self._sp_format = sp_format

    def _get_sparse_format(self):
        return self._sp_format

    def _as_sparse_matrix(self):
        import scipy.sparse as sp_sparse
        if self._sp_operator is None:
            if not self._fill_complete():
                raise ValueError("Not all rows or columns contain operators.")
            for obj in self._operators.ravel():
                assert isinstance(obj, sp_sparse.spmatrix) or obj is None
            self._sp_operator = sp_sparse.bmat(self._operators,
                                               format = self._sp_format)
        return self._sp_operator
    
    def set_empty_entry(self, pos, shape):
        """
        Documentation is missing
        
        TODO: Fix this!
        """
        if self.sp_format is "csr":
            from scipy.sparse import csr_matrix as empty_matrix
        elif self.sp_format is "csc":
            from scipy.sparse import csr_matrix as empty_matrix
        else:
            raise NotImplemented()
        assert isinstance(shape, tuple)
        assert self._operators[pos] is None
        if self._rows[pos[0]] != 0 and self._cols[pos[1]] != 0:
            raise ValueError("Row and column shapes are already set")
        elif self._rows[pos[0]] == 0 and self._cols[pos[1]] != 0:
            self._rows[pos[0]] = shape[0]
            self._operators[pos] = empty_matrix(shape, dtype = self.dtype)
        elif self._rows[pos[0]] != 0 and self._cols[pos[1]] == 0:
            self._cols[pos[1]] = shape[1]
            self._operators[pos] = empty_matrix(shape, dtype = self.dtype)
        elif self._rows[pos[0]] == 0 and self._cols[pos[1]] == 0:
            self._rows[pos[0]] = shape[0]
            self._cols[pos[1]] = shape[1]
            self._operators[pos] = empty_matrix(shape, dtype = self.dtype)
    
    def permute(self, perm):
        """
        Permutation of operator array according to permutation matrix.
        """
        assert self._fill_complete()
        assert isinstance(perm, np.ndarray)
        assert perm.shape[0] == perm.shape[1]
        assert perm.shape == self._operators.shape
        assert np.all(np.sum(perm, axis=0)==1)
        assert np.all(np.sum(perm, axis=1)==1)
        assert not (perm.dot(perm.T)-np.identity(perm.shape[0])).any()
        row_shift = perm.T.nonzero()[1]
        col_shift = perm.nonzero()[1]
        self._rows = self._rows[row_shift]
        self._cols = self._cols[np.argsort(col_shift)]
        self._operators = self._operators[row_shift][:,np.argsort(col_shift)]
        for i in range(self._m):
            for j in range(self._n):
                if self._operators[i, j] is not None:
                    assert self._operators[i,j].shape == (self._rows[i], self._cols[j])
        self._sp_operator = None

    def split_vector(self, x):
        """
        Documentation is missing
        
        TODO: Fix this!
        """
        if not self._fill_complete():
            raise ValueError("Not all rows or columns contain operators.")
        assert x.shape == (np.sum(self._cols), )
        unique_dim = np.unique(self._cols)
        x_map = {}
        row_map = {}
        cnt = {}
        for dim in unique_dim:
            n = np.count_nonzero(self._cols == dim)
            x_map[dim] = np.zeros((dim, n), dtype = x.dtype)
            row_map[dim] = []
            cnt[dim] = 0
        col_dim = 0
        for j in range(self._n):
            local_x = x[col_dim:col_dim + self._cols[j]]
            col_dim += self._cols[j]
            k = cnt[self._cols[j]]
            x_map[self._cols[j]][:,k] = local_x
            cnt[self._cols[j]] += 1
            row_map[self._cols[j]].append(j)
        return x_map, row_map

    shape = property(_get_shape)
    dtype = property(_get_dtype)
    ndims = property(_get_ndims)
    fill_complete = property(_fill_complete)
    row_dimensions = property(_get_row_dimensions)
    column_dimensions = property(_get_column_dimensions)
    sp_format = property(fset = _set_sparse_format, fget = _get_sparse_format)
    sp_operator = property(_as_sparse_matrix)