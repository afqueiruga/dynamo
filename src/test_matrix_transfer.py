# -*- coding: utf-8 -*-
import numpy as np
from scipy.sparse import csr_matrix
m, n, nn = 4, 4, 3

B = np.arange(16).reshape(4,4)
auxB = csr_matrix(B)

rowptr, colind, values = auxB.indptr, auxB.indices, auxB.data
assert rowptr.size == m + 1

mapping = dict()
# switch first two columns
mapping[0] = 1
mapping[1] = 0
# remove 3rd colum
mapping[3] = 2

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
    # delete indices from column and value arrays
    colind = np.delete(colind, del_ind)
    values = np.delete(values, del_ind)
    # correct row pointers
    cnt = len(del_ind)
    rowptr[i+1:] -= cnt

assert colind.max() < nn
modB = csr_matrix((values, colind, rowptr), shape=(m, nn))

print B
print modB.todense()
