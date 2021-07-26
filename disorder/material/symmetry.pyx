#cython: boundscheck=False, wraparound=False, language_level=3

import numpy as np
cimport numpy as np

from cython.parallel import prange

cimport cython

from libc.stdlib cimport qsort
            
cpdef (double, double, double) transform(double x, 
                                         double y, 
                                         double z, 
                                         Py_ssize_t sym, 
                                         Py_ssize_t op) nogil:
    
    if (sym == 0):
        
        if (op == 0): return x,y,z
        elif (op == 1): return -x,-y,z
        elif (op == 2): return -x,y,-z
        elif (op == 3): return x,-y,-z
        elif (op == 4): return z,x,y
        elif (op == 5): return z,-x,-y
        elif (op == 6): return -z,-x,y
        elif (op == 7): return -z,x,-y
        elif (op == 8): return y,z,x
        elif (op == 9): return -y,z,-x
        elif (op == 10): return y,-z,-x
        elif (op == 11): return -y,-z,x
        elif (op == 12): return y,x,-z
        elif (op == 13): return -y,-x,-z
        elif (op == 14): return y,-x,z
        elif (op == 15): return -y,x,z
        elif (op == 16): return x,z,-y
        elif (op == 17): return -x,z,y
        elif (op == 18): return -x,-z,-y
        elif (op == 19): return x,-z,y
        elif (op == 20): return z,y,-x
        elif (op == 21): return z,-y,x
        elif (op == 22): return -z,y,x
        elif (op == 23): return -z,-y,-x
        elif (op == 24): return -x,-y,-z
        elif (op == 25): return x,y,-z
        elif (op == 26): return x,-y,z
        elif (op == 27): return -x,y,z
        elif (op == 28): return -z,-x,-y
        elif (op == 29): return -z,x,y
        elif (op == 30): return z,x,-y
        elif (op == 31): return z,-x,y
        elif (op == 32): return -y,-z,-x
        elif (op == 33): return y,-z,x
        elif (op == 34): return -y,z,x
        elif (op == 35): return y,z,-x
        elif (op == 36): return -y,-x,z
        elif (op == 37): return y,x,z
        elif (op == 38): return -y,x,-z
        elif (op == 39): return y,-x,-z
        elif (op == 40): return -x,-z,y
        elif (op == 41): return x,-z,-y
        elif (op == 42): return x,z,y
        elif (op == 43): return -x,z,-y
        elif (op == 44): return -z,-y,x
        elif (op == 45): return -z,y,-x
        elif (op == 46): return z,-y,-x
        else: return z,y,x
        
    elif (sym == 1):
        
        if (op == 0): return x,y,z
        elif (op == 1): return -x,-y,z
        elif (op == 2): return -x,y,-z
        elif (op == 3): return x,-y,-z
        elif (op == 4): return z,x,y
        elif (op == 5): return z,-x,-y
        elif (op == 6): return -z,-x,y
        elif (op == 7): return -z,x,-y
        elif (op == 8): return y,z,x
        elif (op == 9): return -y,z,-x
        elif (op == 10): return y,-z,-x
        elif (op == 11): return -y,-z,x
        elif (op == 12): return -x,-y,-z
        elif (op == 13): return x,y,-z
        elif (op == 14): return x,-y,z
        elif (op == 15): return -x,y,z
        elif (op == 16): return -z,-x,-y
        elif (op == 17): return -z,x,y
        elif (op == 18): return z,x,-y
        elif (op == 19): return z,-x,y
        elif (op == 20): return -y,-z,-x
        elif (op == 21): return y,-z,x
        elif (op == 22): return -y,z,x
        else: return y,z,-x
        
    elif (sym == 2):
        
        if (op == 0): return x,y,z
        elif (op == 1): return -x-y,x,z
        elif (op == 2): return y,-x-y,z
        elif (op == 3): return -x,-y,z
        elif (op == 4): return x+y,-x,z
        elif (op == 5): return -y,x+y,z
        elif (op == 6): return y,x,-z
        elif (op == 7): return x,-x-y,-z
        elif (op == 8): return -x-y,y,-z
        elif (op == 9): return -y,-x,-z
        elif (op == 10): return -x,x+y,-z
        elif (op == 11): return x+y,-y,-z
        elif (op == 12): return -x,-y,-z
        elif (op == 13): return x+y,-x,-z
        elif (op == 14): return -y,x+y,-z
        elif (op == 15): return x,y,-z
        elif (op == 16): return -x-y,x,-z
        elif (op == 17): return y,-x-y,-z
        elif (op == 18): return -y,-x,z
        elif (op == 19): return -x,x+y,z
        elif (op == 20): return x+y,-y,z
        elif (op == 21): return y,x,z
        elif (op == 22): return x,-x-y,z
        else: return -x-y,y,z

    elif (sym == 3):
            
        if (op == 0): return x,y,z
        elif (op == 1): return -x-y,x,z
        elif (op == 2): return y,-x-y,z
        elif (op == 3): return -x,-y,z
        elif (op == 4): return x+y,-x,z
        elif (op == 5): return -y,x+y,z
        elif (op == 6): return -x,-y,-z
        elif (op == 7): return x+y,-x,-z
        elif (op == 8): return -y,x+y,-z
        elif (op == 9): return x,y,-z
        elif (op == 10): return -x-y,x,-z
        else: return y,-x-y,-z
        
    elif (sym == 4):
        
        if (op == 0): return x,y,z
        elif (op == 1): return -x-y,x,z
        elif (op == 2): return y,-x-y,z
        elif (op == 3): return -y,-x,-z
        elif (op == 4): return -x,x+y,-z
        elif (op == 5): return x+y,-y,-z
        elif (op == 6): return -x,-y,-z
        elif (op == 7): return x+y,-x,-z
        elif (op == 8): return -y,x+y,-z
        elif (op == 9): return y,x,z
        elif (op == 10): return x,-x-y,z
        else: return -x-y,y,z
        
    elif (sym == 5):
   
        if (op == 0): return x,y,z
        elif (op == 1): return -x-y,x,z
        elif (op == 2): return y,-x-y,z
        elif (op == 3): return -x,-y,-z
        elif (op == 4): return x+y,-x,-z
        else: return -y,x+y,-z
        
    elif (sym == 6):
    
        if (op == 0): return x,y,z
        elif (op == 1): return -x,-y,z
        elif (op == 2): return -y,x,z
        elif (op == 3): return y,-x,z
        elif (op == 4): return -x,y,-z
        elif (op == 5): return x,-y,-z
        elif (op == 6): return y,x,-z
        elif (op == 7): return -y,-x,-z
        elif (op == 8): return -x,-y,-z
        elif (op == 9): return x,y,-z
        elif (op == 10): return y,-x,-z
        elif (op == 11): return -y,x,-z
        elif (op == 12): return x,-y,z
        elif (op == 13): return -x,y,z
        elif (op == 14): return -y,-x,z
        else: return y,x,z
        
    elif (sym == 7):
        
        if (op == 0): return x,y,z
        elif (op == 1): return -x,-y,z
        elif (op == 2): return -y,x,z
        elif (op == 3): return y,-x,z
        elif (op == 4): return -x,-y,-z
        elif (op == 5): return x,y,-z
        elif (op == 6): return y,-x,-z
        else: return -y,x,-z
        
    elif (sym == 8):
        
        if (op == 0): return x,y,z
        elif (op == 1): return -x,-y,z
        elif (op == 2): return -x,y,-z
        elif (op == 3): return x,-y,-z
        elif (op == 4): return -x,-y,-z
        elif (op == 5): return x,y,-z
        elif (op == 6): return x,-y,z
        else: return -x,y,z
        
    elif (sym == 9):
        
        if (op == 0): return x,y,z
        elif (op == 1): return -x,y,-z
        elif (op == 2): return -x,-y,-z
        else: return x,-y,z
                
    elif (sym == 10):
        
        if (op == 0): return x,y,z
        else: return -x,-y,-z
        
    else:
        
        return x,y,z
    
cpdef (signed short, signed short, signed short) bragg(signed short x, 
                                                       signed short y, 
                                                       signed short z, 
                                                       signed short sym, 
                                                       signed short op) nogil:
    
    if (sym == 0):
        
        if (op == 0): return x,y,z
        elif (op == 1): return -x,-y,z
        elif (op == 2): return -x,y,-z
        elif (op == 3): return x,-y,-z
        elif (op == 4): return z,x,y
        elif (op == 5): return z,-x,-y
        elif (op == 6): return -z,-x,y
        elif (op == 7): return -z,x,-y
        elif (op == 8): return y,z,x
        elif (op == 9): return -y,z,-x
        elif (op == 10): return y,-z,-x
        elif (op == 11): return -y,-z,x
        elif (op == 12): return y,x,-z
        elif (op == 13): return -y,-x,-z
        elif (op == 14): return y,-x,z
        elif (op == 15): return -y,x,z
        elif (op == 16): return x,z,-y
        elif (op == 17): return -x,z,y
        elif (op == 18): return -x,-z,-y
        elif (op == 19): return x,-z,y
        elif (op == 20): return z,y,-x
        elif (op == 21): return z,-y,x
        elif (op == 22): return -z,y,x
        elif (op == 23): return -z,-y,-x
        elif (op == 24): return -x,-y,-z
        elif (op == 25): return x,y,-z
        elif (op == 26): return x,-y,z
        elif (op == 27): return -x,y,z
        elif (op == 28): return -z,-x,-y
        elif (op == 29): return -z,x,y
        elif (op == 30): return z,x,-y
        elif (op == 31): return z,-x,y
        elif (op == 32): return -y,-z,-x
        elif (op == 33): return y,-z,x
        elif (op == 34): return -y,z,x
        elif (op == 35): return y,z,-x
        elif (op == 36): return -y,-x,z
        elif (op == 37): return y,x,z
        elif (op == 38): return -y,x,-z
        elif (op == 39): return y,-x,-z
        elif (op == 40): return -x,-z,y
        elif (op == 41): return x,-z,-y
        elif (op == 42): return x,z,y
        elif (op == 43): return -x,z,-y
        elif (op == 44): return -z,-y,x
        elif (op == 45): return -z,y,-x
        elif (op == 46): return z,-y,-x
        else: return z,y,x
        
    elif (sym == 1):
        
        if (op == 0): return x,y,z
        elif (op == 1): return -x,-y,z
        elif (op == 2): return -x,y,-z
        elif (op == 3): return x,-y,-z
        elif (op == 4): return z,x,y
        elif (op == 5): return z,-x,-y
        elif (op == 6): return -z,-x,y
        elif (op == 7): return -z,x,-y
        elif (op == 8): return y,z,x
        elif (op == 9): return -y,z,-x
        elif (op == 10): return y,-z,-x
        elif (op == 11): return -y,-z,x
        elif (op == 12): return -x,-y,-z
        elif (op == 13): return x,y,-z
        elif (op == 14): return x,-y,z
        elif (op == 15): return -x,y,z
        elif (op == 16): return -z,-x,-y
        elif (op == 17): return -z,x,y
        elif (op == 18): return z,x,-y
        elif (op == 19): return z,-x,y
        elif (op == 20): return -y,-z,-x
        elif (op == 21): return y,-z,x
        elif (op == 22): return -y,z,x
        else: return y,z,-x
        
    elif (sym == 2):
        
        if (op == 0): return x,y,z
        elif (op == 1): return -x-y,x,z
        elif (op == 2): return y,-x-y,z
        elif (op == 3): return -x,-y,z
        elif (op == 4): return x+y,-x,z
        elif (op == 5): return -y,x+y,z
        elif (op == 6): return y,x,-z
        elif (op == 7): return x,-x-y,-z
        elif (op == 8): return -x-y,y,-z
        elif (op == 9): return -y,-x,-z
        elif (op == 10): return -x,x+y,-z
        elif (op == 11): return x+y,-y,-z
        elif (op == 12): return -x,-y,-z
        elif (op == 13): return x+y,-x,-z
        elif (op == 14): return -y,x+y,-z
        elif (op == 15): return x,y,-z
        elif (op == 16): return -x-y,x,-z
        elif (op == 17): return y,-x-y,-z
        elif (op == 18): return -y,-x,z
        elif (op == 19): return -x,x+y,z
        elif (op == 20): return x+y,-y,z
        elif (op == 21): return y,x,z
        elif (op == 22): return x,-x-y,z
        else: return -x-y,y,z

    elif (sym == 3):
            
        if (op == 0): return x,y,z
        elif (op == 1): return -x-y,x,z
        elif (op == 2): return y,-x-y,z
        elif (op == 3): return -x,-y,z
        elif (op == 4): return x+y,-x,z
        elif (op == 5): return -y,x+y,z
        elif (op == 6): return -x,-y,-z
        elif (op == 7): return x+y,-x,-z
        elif (op == 8): return -y,x+y,-z
        elif (op == 9): return x,y,-z
        elif (op == 10): return -x-y,x,-z
        else: return y,-x-y,-z
        
    elif (sym == 4):
        
        if (op == 0): return x,y,z
        elif (op == 1): return -x-y,x,z
        elif (op == 2): return y,-x-y,z
        elif (op == 3): return -y,-x,-z
        elif (op == 4): return -x,x+y,-z
        elif (op == 5): return x+y,-y,-z
        elif (op == 6): return -x,-y,-z
        elif (op == 7): return x+y,-x,-z
        elif (op == 8): return -y,x+y,-z
        elif (op == 9): return y,x,z
        elif (op == 10): return x,-x-y,z
        else: return -x-y,y,z
        
    elif (sym == 5):
   
        if (op == 0): return x,y,z
        elif (op == 1): return -x-y,x,z
        elif (op == 2): return y,-x-y,z
        elif (op == 3): return -x,-y,-z
        elif (op == 4): return x+y,-x,-z
        else: return -y,x+y,-z
        
    elif (sym == 6):
    
        if (op == 0): return x,y,z
        elif (op == 1): return -x,-y,z
        elif (op == 2): return -y,x,z
        elif (op == 3): return y,-x,z
        elif (op == 4): return -x,y,-z
        elif (op == 5): return x,-y,-z
        elif (op == 6): return y,x,-z
        elif (op == 7): return -y,-x,-z
        elif (op == 8): return -x,-y,-z
        elif (op == 9): return x,y,-z
        elif (op == 10): return y,-x,-z
        elif (op == 11): return -y,x,-z
        elif (op == 12): return x,-y,z
        elif (op == 13): return -x,y,z
        elif (op == 14): return -y,-x,z
        else: return y,x,z
        
    elif (sym == 7):
        
        if (op == 0): return x,y,z
        elif (op == 1): return -x,-y,z
        elif (op == 2): return -y,x,z
        elif (op == 3): return y,-x,z
        elif (op == 4): return -x,-y,-z
        elif (op == 5): return x,y,-z
        elif (op == 6): return y,-x,-z
        else: return -y,x,-z
        
    elif (sym == 8):
        
        if (op == 0): return x,y,z
        elif (op == 1): return -x,-y,z
        elif (op == 2): return -x,y,-z
        elif (op == 3): return x,-y,-z
        elif (op == 4): return -x,-y,-z
        elif (op == 5): return x,y,-z
        elif (op == 6): return x,-y,z
        else: return -x,y,z
        
    elif (sym == 9):
        
        if (op == 0): return x,y,z
        elif (op == 1): return -x,y,-z
        elif (op == 2): return -x,-y,-z
        else: return x,-y,z
                
    elif (sym == 10):
        
        if (op == 0): return x,y,z
        else: return -x,-y,-z
        
    else:
        
        return x,y,z
    
cdef int compare(const void *p, const void *q) nogil:
        
    cdef signed short *x = (<signed short*>p)
    cdef signed short *y = (<signed short*>q)
    
    cdef signed short diff
    
    diff = x[2] - y[2]
    if (diff):
        return diff
    diff = x[1] - y[1]
    if (diff):
        return diff
    return x[0] - y[0]

cdef void colsort(signed short [:,::1] arr) nogil:

    qsort(&arr[0,0], arr.shape[0], arr.strides[0], &compare)
    
cpdef void friedel(signed short [:,::1] pair,
                   signed short [:,:,::1] coordinate):
    
    cdef Py_ssize_t n = coordinate.shape[0]
    
    p_np = np.zeros((2,3), dtype=np.int16)
    
    cdef signed short [:,::1] p = p_np

    cdef Py_ssize_t i, j, k
  
    with nogil:
        
        for i in range(n):
            for j in range(2):
                for k in range(3):
                
                    p[j,k] = coordinate[i,j,k]
                
            colsort(p)
                
            for k in range(3):
            
                pair[i,k] = p[0,k]
    
cpdef void sorting(signed short [:,::1] total, 
                   signed short [:,::1] cosymmetries, 
                   symop):
    
    cdef Py_ssize_t n_hkl = total.shape[0]
    
    cdef Py_ssize_t sym = symop[0]
    cdef Py_ssize_t n_ops = symop[1]
    
    ops_np = np.zeros((n_ops,3), dtype=np.int16)
    
    cdef signed short [:,::1] ops = ops_np
    
    cdef Py_ssize_t i, op 
    
    cdef signed short h, k, l, h_, k_, l_
    
    for i in range(n_hkl):

        h_, k_, l_ = cosymmetries[i,0], cosymmetries[i,1], cosymmetries[i,2]

        for op in range(n_ops):    
                                
            h, k, l = bragg(h_, k_, l_, sym, op)
            
            ops[op,0], ops[op,1], ops[op,2] = h, k, l
            
        colsort(ops)
                
        total[i,0], total[i,1], total[i,2] = ops[0,0], ops[0,1], ops[0,2]
        
def unique(data):
    
    b = np.ascontiguousarray(data).view(np.dtype((np.void, 
                                           data.dtype.itemsize*data.shape[1])))
    u, ind, inv = np.unique(b, return_index=True, return_inverse=True)
    
    return u.view(data.dtype).reshape(-1, data.shape[1]), ind, inv
        
def evaluate(operator, coordinates, translate=True):
    
    operator = str(operator)
    
    x, y, z = coordinates
        
    if (not translate):
        operator = operator.replace('/','//')       
                
    return np.array(eval(operator))

def evaluate_mag(operator, moments):
   
    operator = str(operator)

    mx, my, mz = moments
                
    return np.array(eval(operator))

def reverse(symops):
    
    if (type(symops) == str or type(symops) == np.str_):
        symops = [symops]
    
    uvw = np.array(['x','y','z'])

    rsymops = []
        
    for symop in symops:

        symop = str(symop)

        W = np.zeros((3,3))
            
        W[:,0] = evaluate(symop, [1,0,0], translate=False)
        W[:,1] = evaluate(symop, [0,1,0], translate=False)
        W[:,2] = evaluate(symop, [0,0,1], translate=False)
        
        w = evaluate(symop, [0,0,0], translate=True)
                
        W_inv = np.linalg.inv(W).round()
        w_inv = -np.dot(W_inv, w)  

        W_inv = W_inv.astype(int).astype(str)
        w_inv = [str(float.as_integer_ratio(c)).replace('(', '')\
                                               .replace(')', '')\
                                               .replace(', ', '/')\
                                               .replace('0/1', '')\
                                               for c in w_inv]

        rop = u''
                
        for i in range(3):
            string = ''
            for j in range(3):
                if (W_inv[i,j] == '-1'):
                    string += '-'+uvw[j]
                elif (W_inv[i,j] == '1'):
                    if (len(string) == 0):
                        string += uvw[j]
                    else:
                        string += '+'+uvw[j]
            if (len(w_inv[i]) > 0):
                if (w_inv[i][0] == '-'):
                    string += w_inv[i]
                else:
                    string += '+'+w_inv[i]
            rop += string
            if (i != 2):
                rop += ','
                                    
        rsymops.append(rop)
        
    return np.array(rsymops)

def inverse(symops):
    
    if (type(symops) == str or type(symops) == np.str_):
        symops = [symops]
    
    uvw = np.array(['x','y','z'])

    rsymops = []
        
    for symop in symops:
        
        symop = str(symop)
        
        W = np.zeros((3,3))
            
        W[:,0] = evaluate(symop, [1,0,0], translate=False)
        W[:,1] = evaluate(symop, [0,1,0], translate=False)
        W[:,2] = evaluate(symop, [0,0,1], translate=False)
        
        W_inv = np.linalg.inv(W).T.round().astype(int).astype(str)
        
        rop = u''
        
        for i in range(3):
            string = ''
            for j in range(3):
                if (W_inv[i,j] == '-1'):
                    string += '-'+uvw[j]
                elif (W_inv[i,j] == '1'):
                    if (len(string) == 0):
                        string += uvw[j]
                    else:
                        string += '+'+uvw[j]
            rop += string
            if (i != 2):
                rop += ','
                                    
        rsymops.append(rop)
        
    return np.array(rsymops)

def binary(symop0, symop1):
    
    symop0 = str(symop0)
    symop1 = str(symop1)
    
    uvw = np.array(['x','y','z'])
        
    W0, W1 = np.zeros((3,3)), np.zeros((3,3))
    
    w0 = evaluate(symop0, [0,0,0], translate=True)
    w1 = evaluate(symop1, [0,0,0], translate=True)
            
    W0[:,0] = evaluate(symop0, [1,0,0], translate=False)
    W0[:,1] = evaluate(symop0, [0,1,0], translate=False)
    W0[:,2] = evaluate(symop0, [0,0,1], translate=False)
    
    W1[:,0] = evaluate(symop1, [1,0,0], translate=False)
    W1[:,1] = evaluate(symop1, [0,1,0], translate=False)
    W1[:,2] = evaluate(symop1, [0,0,1], translate=False)
    
    W = np.dot(W0, W1).round().astype(int).astype(str)
    
    w = np.dot(W0, w1)+w0
    
    w = [str(float.as_integer_ratio(c)).replace('(', '')\
                                       .replace(')', '')\
                                       .replace(', ', '/')\
                                       .replace('0/1', '') for c in w]
        
    symop = u''
    
    for i in range(3):
        string = ''
        for j in range(3):
            if (W[i,j] == '-1'):
                string += '-'+uvw[j]
            elif (W[i,j] == '1'):
                if (len(string) == 0):
                    string += uvw[j]
                else:
                    string += '+'+uvw[j]
        if (len(w[i]) > 0):
            if (w[i][0] == '-'):
                string += w[i]
            else:
                string += '+'+w[i]
        symop += string
        if (i != 2):
            symop += ','
            
    return symop

def operators(invert=False):
    
    laue = {
    
    'm-3m' : [u'x,y,z',u'-x,-y,z',u'-x,y,-z',u'x,-y,-z',
              u'z,x,y',u'z,-x,-y',u'-z,-x,y',u'-z,x,-y',
              u'y,z,x',u'-y,z,-x',u'y,-z,-x',u'-y,-z,x',
              u'y,x,-z',u'-y,-x,-z',u'y,-x,z',u'-y,x,z',
              u'x,z,-y',u'-x,z,y',u'-x,-z,-y',u'x,-z,y',
              u'z,y,-x',u'z,-y,x',u'-z,y,x',u'-z,-y,-x',
              u'-x,-y,-z',u'x,y,-z',u'x,-y,z',u'-x,y,z',
              u'-z,-x,-y',u'-z,x,y',u'z,x,-y',u'z,-x,y',
              u'-y,-z,-x',u'y,-z,x',u'-y,z,x',u'y,z,-x',
              u'-y,-x,z',u'y,x,z',u'-y,x,-z',u'y,-x,-z',
              u'-x,-z,y',u'x,-z,-y',u'x,z,y',u'-x,z,-y',
              u'-z,-y,x',u'-z,y,-x',u'z,-y,-x',u'z,y,x'],
        
    'm-3' : [u'x,y,z',u'-x,-y,z',u'-x,y,-z',u'x,-y,-z',
             u'z,x,y',u'z,-x,-y',u'-z,-x,y',u'-z,x,-y',
             u'y,z,x',u'-y,z,-x',u'y,-z,-x',u'-y,-z,x',
             u'-x,-y,-z',u'x,y,-z',u'x,-y,z',u'-x,y,z',
             u'-z,-x,-y',u'-z,x,y',u'z,x,-y',u'z,-x,y',
             u'-y,-z,-x',u'y,-z,x',u'-y,z,x',u'y,z,-x'],
        
    '6/mmm' : [u'x,y,z',u'-y,x-y,z',u'-x+y,-x,z',u'-x,-y,z',
               u'y,-x+y,z',u'x-y,x,z',u'y,x,-z',u'x-y,-y,-z',
               u'-x,-x+y,-z',u'-y,-x,-z',u'-x+y,y,-z',u'x,x-y,-z',
               u'-x,-y,-z',u'y,-x+y,-z',u'x-y,x,-z',u'x,y,-z',
               u'-y,x-y,-z',u'-x+y,-x,-z',u'-y,-x,z',u'-x+y,y,z',
               u'x,x-y,z',u'y,x,z',u'x-y,-y,z',u'-x,-x+y,z'],
        
    '6/m' : [u'x,y,z',u'-y,x-y,z',u'-x+y,-x,z',u'-x,-y,z',
             u'y,-x+y,z',u'x-y,x,z',u'-x,-y,-z',u'y,-x+y,-z',
             u'x-y,x,-z',u'x,y,-z',u'-y,x-y,-z',u'-x+y,-x,-z'],
        
    '-3m' : [u'x,y,z',u'-y,x-y,z',u'-x+y,-x,z',u'-y,-x,-z',
             u'-x+y,y,-z',u'x,x-y,-z',u'-x,-y,-z',u'y,-x+y,-z',
             u'x-y,x,-z',u'y,x,z',u'x-y,-y,z',u'-x,-x+y,z'],
        
    '-3' : [u'x,y,z',u'-y,x-y,z',u'-x+y,-x,z',
            u'-x,-y,-z',u'y,-x+y,-z',u'x-y,x,-z'],
        
    '4/mmm' : [u'x,y,z',u'-x,-y,z',u'-y,x,z',u'y,-x,z',
               u'-x,y,-z',u'x,-y,-z',u'y,x,-z',u'-y,-x,-z',
               u'-x,-y,-z',u'x,y,-z',u'y,-x,-z',u'-y,x,-z',
               u'x,-y,z',u'-x,y,z',u'-y,-x,z',u'y,x,z'],
        
    '4/m' : [u'x,y,z',u'-x,-y,z',u'-y,x,z',u'y,-x,z',
             u'-x,-y,-z',u'x,y,-z',u'y,-x,-z',u'-y,x,-z'],
        
    'mmm' : [u'x,y,z',u'-x,-y,z',u'-x,y,-z',u'x,-y,-z',
             u'-x,-y,-z',u'x,y,-z',u'x,-y,z',u'-x,y,z'],
    
    '2/m': [u'x,y,z',u'-x,y,-z',u'-x,-y,-z',u'x,-y,z'],
        
    '-1' : [u'x,y,z',u'-x,-y,-z']
    
    }
    
    if (invert):
    
        symmetry = list(laue.keys())
                
        for sym in symmetry:

            laue[sym] = inverse(laue.get(sym)).tolist()
        
    return laue
        
def laue(symmetry):
    
    if (symmetry == 'm-3m'):
        
        ops = [u'x,y,z',u'-x,-y,z',u'-x,y,-z',u'x,-y,-z',
               u'z,x,y',u'z,-x,-y',u'-z,-x,y',u'-z,x,-y',
               u'y,z,x',u'-y,z,-x',u'y,-z,-x',u'-y,-z,x',
               u'y,x,-z',u'-y,-x,-z',u'y,-x,z',u'-y,x,z',
               u'x,z,-y',u'-x,z,y',u'-x,-z,-y',u'x,-z,y',
               u'z,y,-x',u'z,-y,x',u'-z,y,x',u'-z,-y,-x',
               u'-x,-y,-z',u'x,y,-z',u'x,-y,z',u'-x,y,z',
               u'-z,-x,-y',u'-z,x,y',u'z,x,-y',u'z,-x,y',
               u'-y,-z,-x',u'y,-z,x',u'-y,z,x',u'y,z,-x',
               u'-y,-x,z',u'y,x,z',u'-y,x,-z',u'y,-x,-z',
               u'-x,-z,y',u'x,-z,-y',u'x,z,y',u'-x,z,-y',
               u'-z,-y,x',u'-z,y,-x',u'z,-y,-x',u'z,y,x']
        
    elif (symmetry == 'm-3'):
        
        ops = [u'x,y,z',u'-x,-y,z',u'-x,y,-z',u'x,-y,-z',
               u'z,x,y',u'z,-x,-y',u'-z,-x,y',u'-z,x,-y',
               u'y,z,x',u'-y,z,-x',u'y,-z,-x',u'-y,-z,x',
               u'-x,-y,-z',u'x,y,-z',u'x,-y,z',u'-x,y,z',
               u'-z,-x,-y',u'-z,x,y',u'z,x,-y',u'z,-x,y',
               u'-y,-z,-x',u'y,-z,x',u'-y,z,x',u'y,z,-x']
        
    elif (symmetry == '6/mmm'):
        
        ops = [u'x,y,z',u'-y,x-y,z',u'-x+y,-x,z',u'-x,-y,z',
               u'y,-x+y,z',u'x-y,x,z',u'y,x,-z',u'x-y,-y,-z',
               u'-x,-x+y,-z',u'-y,-x,-z',u'-x+y,y,-z',u'x,x-y,-z',
               u'-x,-y,-z',u'y,-x+y,-z',u'x-y,x,-z',u'x,y,-z',
               u'-y,x-y,-z',u'-x+y,-x,-z',u'-y,-x,z',u'-x+y,y,z',
               u'x,x-y,z',u'y,x,z',u'x-y,-y,z',u'-x,-x+y,z']
        
    elif (symmetry == '6/m'):
        
        ops = [u'x,y,z',u'-y,x-y,z',u'-x+y,-x,z',u'-x,-y,z',
               u'y,-x+y,z',u'x-y,x,z',u'-x,-y,-z',u'y,-x+y,-z',
               u'x-y,x,-z',u'x,y,-z',u'-y,x-y,-z',u'-x+y,-x,-z']
        
    elif (symmetry == '-3m'):
        
        ops = [u'x,y,z',u'-y,x-y,z',u'-x+y,-x,z',u'-y,-x,-z',
               u'-x+y,y,-z',u'x,x-y,-z',u'-x,-y,-z',u'y,-x+y,-z',
               u'x-y,x,-z',u'y,x,z',u'x-y,-y,z',u'-x,-x+y,z']
        
    elif (symmetry == '-3'):
        
        ops = [u'x,y,z',u'-y,x-y,z',u'-x+y,-x,z',
               u'-x,-y,-z',u'y,-x+y,-z',u'x-y,x,-z']
        
    elif (symmetry == '4/mmm'):
    
        ops = [u'x,y,z',u'-x,-y,z',u'-y,x,z',u'y,-x,z',
               u'-x,y,-z',u'x,-y,-z',u'y,x,-z',u'-y,-x,-z',
               u'-x,-y,-z',u'x,y,-z',u'y,-x,-z',u'-y,x,-z',
               u'x,-y,z',u'-x,y,z',u'-y,-x,z',u'y,x,z']
        
    elif (symmetry == '4/m'):
        
        ops = [u'x,y,z',u'-x,-y,z',u'-y,x,z',u'y,-x,z',
               u'-x,-y,-z',u'x,y,-z',u'y,-x,-z',u'-y,x,-z']
        
    elif (symmetry == 'mmm'):
        
        ops = [u'x,y,z',u'-x,-y,z',u'-x,y,-z',u'x,-y,-z',
               u'-x,-y,-z',u'x,y,-z',u'x,-y,z',u'-x,y,z']
        
    elif (symmetry == '2/m'):
        
        ops = [u'x,y,z',u'-x,y,-z',u'-x,-y,-z',u'x,-y,z']
                
    elif (symmetry == '-1'):
        
        ops = [u'x,y,z',u'-x,-y,-z']
        
    else:
        
        ops = [u'x,y,z']
        
    return ops