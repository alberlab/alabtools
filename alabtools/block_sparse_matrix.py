from scipy.sparse import csr_matrix, bmat
from scipy.sparse.sputils import isshape, ismatrix, issequence
from collections import defaultdict
import h5py
import numpy as np
import math

class h5_csr_matrix(csr_matrix):
    '''
    A csr matrix on h5 file. Read only
    '''
    def __init__(self, arg1, lazy=True, mode='r', out=None, **kwargs):
        self.h5 = None
        self.mode = mode
        self.attrs = ['shape']

        if isinstance(arg1, str):
            self.h5 = h5py.File(arg1, mode=mode)

        if isinstance(arg1, h5py.Group):
            self.h5 = arg1

            if mode in ['r', 'r+', 'a']:
                super().__init__(self.h5['shape'])

                if lazy:
                    for k in self.h5.keys():
                        self.attrs.append(k)
                        if k == 'shape':
                            continue
                        self.__setattr__(k, self.h5[k])
                else:
                    for k in self.h5.keys():
                        self.attrs.append(k)
                        if k == 'shape':
                            continue
                        self.__setattr__(k, self.h5[k][:])
                    if isinstance(self.h5, h5py.File):
                        self.h5.close()
                        self.h5 = None
            else:
                raise ValueError('to open in write mode, instantiate as a csr_matrix (or with an existing h5_csr_matrix) and use the keyword argument `out`')

        elif mode in ['w']:
            self.attrs += ['data', 'indices', 'indptr']
            super().__init__(arg1, **kwargs)
            self.h5 = h5py.File(out, mode=mode)

    def __del__(self):
        self.close()

    def __setattr__(self, key, value):
        if key not in self.attrs:
            self.attrs.append(key)
        return super().__setattr__(key, value)

    def write(self, arg1=None):

        if arg1 is None:
            arg1 = self.h5

        if isinstance(arg1, str):
            with h5py.File(arg1, mode='r+') as h5:
                for k in self.attrs:
                    if k in h5:
                        h5[k][...] = self.__getattr__(k)
                    else:
                        h5.create_dataset(k, self.__getattr__(k))

        elif isinstance(arg1, h5py.Group):
            for k in self.attrs:
                if k in arg1:
                    arg1[k][...] = self.__getattr__(k)
                else:
                    arg1.create_dataset(k, self.__getattr__(k))

    def close(self):
        if isinstance(self.h5, h5py.File):
            if self.mode in ['w', 'r+', 'a']:
                self.write()
            self.h5.close()

class BlockCsrMatrix:
    '''
    Instantiate a block csr matrix

    Parameters
    ----------
        arg1: shape or h5 filename
        N, M: int
            number of rows/columns of blocks
    '''
    def __init__(self, arg1, N=None, M=None, mode='r', dtype=np.float32, lazy=True):

        self.h5 = None
        self.diagonal = None

        if isshape(arg1):
            self.shape = arg1
            self.N = N
            self.M = M
            self.nbr = int(math.ceil(arg1[0] / self.N))
            self.nbc = int(math.ceil(arg1[1] / self.M))
            self.blocks = [
                [
                    csr_matrix(
                        (
                            min(N, self.shape[0] - i*self.N),
                            min(M, self.shape[1] - j*self.M),
                        ), dtype=dtype) for j in range(self.nbc)
                ] for i in range(self.nbr)
            ]

        elif isinstance(arg1, str):
            self.h5 = h5py.File(arg1, mode=mode)

        elif isinstance(arg1, h5py.Group):
            self.h5 = arg1

        if isinstance(self.h5, h5py.Group):
            self.shape = self.h5['shape'][:]
            self.N = self.h5['N']
            self.M = self.h5['M']
            self.nbr = int(math.ceil(arg1[0] / self.N))
            self.nbc = int(math.ceil(arg1[1] / self.M))
            self.blocks = [
                [
                    h5_csr_matrix(self.h5['blocks']['{}_{}'.format(i, j)], lazy=lazy) for j in range(self.nbc)
                ] for i in range(self.nbr)
            ]
            self.diagonal = self.h5['diagonal'][:]


    def _get_len(self, obj, i, N, L):
        if hasattr(obj, '__len__'):
            return len(obj)
        elif isinstance(obj, slice):
            start, stop, step = obj.start, obj.stop, obj.step
            if start is None:
                start = i*N
            else:
                start += i*N
            if stop is None:
                stop = min((i + 1) * N, L)
            if step is None:
                step = 1
            return ( (stop - start - 1) // step ) + 1

    @staticmethod
    def _split_slice(spl, N, L):

        start, stop, step = spl.start, spl.stop, spl.step
        if start is None:
            start = 0
        bstart = start // N

        if stop is None:
            stop = L
        bstop = int(math.ceil(stop / N))

        if step is None:
            step = 1


        items = []
        z = start
        zoffs = (start % N) % step
        k = step * (N // step + 1) - N

        bi = bstart
        while bi < bstop:
            ss = z % N
            if stop >= N * (bi + 1) or stop >= L:
                st = None
            else:
                st = stop % N

            if ss is None or st is None or ss < st:
                items.append((bi, slice(ss, st, step)))

            if step > N:
                z += step
                bi = z // N
            else:
                bi += 1
                zoffs = (zoffs + k) % step
                z = bi * N + zoffs

        return items

    @staticmethod
    def _split_list(lst, N):

        items = defaultdict(set)
        for i, v in enumerate(lst):
            items[v // N].add(v % N)

        return [ (k, list(sorted(v))) for k, v in items.items() ]

    def _fetch(self, rows=None, cols=None):

        if rows is None:
            rows = slice(None, None, None)
        if isinstance(rows, (int, np.int)):
            rows = [rows]
        if cols is None:
            cols = slice(None, None, None)
        if isinstance(cols, (int, np.int)):
            cols = [cols]

        c_el, r_el, c_map, r_map = [None] * 4

        if isinstance(rows, slice):
            r_el = self._split_hslice(rows)
        elif isinstance(rows, list):
            r_el = self._split_list(rows, self.N)
            r_map = {}
            z = 0
            for k, v in r_el:
                for i, x in enumerate(v):
                    r_map[k*self.N + x] = z
                    z += 1

        if isinstance(cols, slice):
            c_el = self._split_vslice(cols)
        elif isinstance(cols, list):
            c_el = self._split_list(cols, self.M)
            c_map = {}
            z = 0
            for k, v in c_el:
                for i, x in enumerate(v):
                    c_map[k * self.M + x] = z
                    z += 1

        mats = [None] * len(r_el)
        for i, rs in r_el:
            mats[i] = [None] * len(c_el)
            for j, cs in c_el:
                if isinstance(rs, list) and isinstance(cs, list):
                    mats[i][j] = self.blocks[i][j][np.ix_(rs, cs)]
                else:
                    mats[i][j] = self.blocks[i][j][rs, cs]

        print(r_el, c_el, mats)

        B = bmat(mats, format='csr')
        if r_map:
            B = B[[r_map[x] for x in rows]]
        if c_map:
            B = B[:, [c_map[x] for x in cols]]
        return B

    def _set(self, rows=None, cols=None, values=None):

        values = np.array(values)

        if rows is None:
            rows = slice(None, None, None)
        if isinstance(rows, (int, np.int)):
            rows = [rows]
        if cols is None:
            cols = slice(None, None, None)
        if isinstance(cols, (int, np.int)):
            cols = [cols]


        c_el, r_el, c_map, r_map = [None] * 4

        if isinstance(rows, slice):
            r_el = self._split_hslice(rows)
        elif isinstance(rows, list):
            r_el = self._split_list(rows, self.N)
            r_map = {}
            z = 0
            for k, v in r_el:
                for i, x in enumerate(v):
                    r_map[k*self.N + x] = z
                    z += 1

        if isinstance(cols, slice):
            c_el = self._split_vslice(cols)
        elif isinstance(cols, list):
            c_el = self._split_list(cols, self.M)
            c_map = {}
            z = 0
            for k, v in c_el:
                for i, x in enumerate(v):
                    c_map[k * self.M + x] = z
                    z += 1

        n_rows, n_cols = 0, 0
        for i, rs in r_el:
            n_rows += self._get_len(rs, i, self.N, self.shape[0])
        for j, cs in c_el:
            n_cols += self._get_len(cs, j, self.M, self.shape[1])

        if ismatrix(values):
            if values.size == 1:
                values = values[0][0]
            elif values.shape[0] != n_rows or values.shape[1] != n_cols:
                raise ValueError(
                    'incompatible shapes: {} x {} vs {} x {}'.format(n_rows, n_cols, values.shape[0], values.shape[1]))
        elif issequence(values):
            if values.size == 1:
                values = values[0]
            elif len(values) != n_rows * n_cols:
                raise ValueError('incompatible shapes: {} x {}, 1 x {}'.format(n_rows, n_cols, len(values)))

        offs1 = 0
        for i, rs in r_el:
            offs2 = 0
            h = self._get_len(rs, i, self.N, self.shape[0])
            for j, cs in c_el:
                if isinstance(rs, list) and isinstance(cs, list):
                    ind = np.ix_(rs, cs)
                else:
                    ind = rs, cs

                w = self._get_len(cs, j, self.M, self.shape[1])

                if ismatrix(values):
                    self.blocks[i][j][ind] = values[offs1:offs1 + h, offs2:offs2 + w]

                elif issequence(values):
                    if n_rows == 1 and n_cols == 1:
                        self.blocks[i][j][ind] = values[0]
                    elif n_rows == 1:
                        self.blocks[i][j][ind] = values[offs1:offs1 + h]
                    elif n_cols == 1:
                        self.blocks[i][j][ind] = values[offs2:offs2 + w]

                else:
                    self.blocks[i][j][ind] = values

                offs2 += w
            offs1 += h

        return values

    def _split_hslice(self, s):
        return self._split_slice(s, self.N, self.shape[0])

    def _split_vslice(self, s):
        return self._split_slice(s, self.M, self.shape[1])

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return self._fetch(item[0], item[1])
        else:
            return self._fetch(item)

    def __setitem__(self, item, values):
        if isinstance(item, tuple):
            return self._set(item[0], item[1], values=values)
        else:
            return self._set(item, values=values)

    def tosparse(self, format='csr'):
        return bmat(self.blocks, format=format)

    def toarray(self):
        return bmat(self.blocks).toarray()


class BlockSymmetricCsrMatrix(BlockCsrMatrix):

    def _fetch(self, rows=None, cols=None):

        if rows is None:
            rows = slice(None, None, None)
        if isinstance(rows, (int, np.int)):
            rows = [rows]
        if cols is None:
            cols = slice(None, None, None)
        if isinstance(cols, (int, np.int)):
            cols = [cols]

        c_el, r_el, c_map, r_map = [None] * 4

        if isinstance(rows, slice):
            r_el = self._split_hslice(rows)
        elif isinstance(rows, list):
            r_el = self._split_list(rows, self.N)
            r_map = {}
            z = 0
            for k, v in r_el:
                for i, x in enumerate(v):
                    r_map[k * self.N + x] = z
                    z += 1

        if isinstance(cols, slice):
            c_el = self._split_vslice(cols)
        elif isinstance(cols, list):
            c_el = self._split_list(cols, self.M)
            c_map = {}
            z = 0
            for k, v in c_el:
                for i, x in enumerate(v):
                    c_map[k * self.M + x] = z
                    z += 1

        mats = [None] * len(r_el)
        for i, rs in r_el:
            mats[i] = [None] * len(c_el)
            for j, cs in c_el:
                if j < i:
                    continue
                else:
                    if isinstance(rs, list) and isinstance(cs, list):
                        mats[i][j] = self.blocks[i][j][np.ix_(rs, cs)]
                        T = mats[i][j].T
                    else:
                        mats[i][j] = self.blocks[i][j][rs, cs]
                        T = mats[i][j].T
                    if i == j:
                        mats[i][j] += T
                    else:
                        mats[j][i] = T

        B = bmat(mats, format='csr')
        if r_map:
            B = B[[r_map[x] for x in rows]]
        if c_map:
            B = B[:, [c_map[x] for x in cols]]
        return B




