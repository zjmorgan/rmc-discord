#!/usr/bin/env python3

import re

import numpy as np

def bragg(x, y, z, sym, op):

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
    
def fraction(x):

    if (x >= 0):
        sign = ''
    else:
        sign = '-'
    
    x = abs(x)
    
    q, r = int(x // 1), x % 1
        
    if (r < 0.47):
        if (r < 0.24):
            if (r < 0.16):
                if (r < 0.12):
                    if (r < 0.11):
                        if (r < 0.09):
                            f = str(q) # 0.0
                        else:
                            f = str(q*10+1)+'/10' # 0.1
                    else:
                        f = str(q*9+1)+'/9' # 0.1111....
                else:
                    if (r < 0.14):
                        f = str(q*8+1)+'/8' # 0.125
                    else:
                        f = str(q*7+1)+'/7' # 0.1428...
            else:
                if (r < 0.19):
                    f = str(q*6+1)+'/6' # 0.1666...
                else:
                    if (r < 0.22):
                        f = str(q*5+1)+'/5' # 0.2
                    else:
                        f = str(q*9+2)+'/9' # 0.2222...
        else:
            if (r < 0.37): 
                if (r < 0.28): 
                    f = str(q*4+1)+'/4' # 0.25
                else:
                    if (r < 0.31):
                        f = str(q*7+2)+'/7' # 0.2857...
                    else:
                        f = str(q*3+1)+'/3' # 0.3333...
            else:
                if (r < 0.42): 
                    if (r < 0.40):
                        f = str(q*8+3)+'/8' # 0.375
                    else:
                        f = str(q*5+2)+'/5' # 0.4
                else:
                    if (r < 0.44):
                        f = str(q*7+3)+'/7' # 0.4285...
                    else:
                        f = str(q*9+4)+'/9' # 0.4444...
    else:
        if (r < 0.71):
            if (r < 0.60):
                if (r < 0.55):
                    f = str(q*2+1)+'/2' # 0.5
                else:
                    if (r < 0.57):
                        f = str(q*9+5)+'/9' # 0.5555...
                    else:
                        f = str(q*7+4)+'/7' # 0.5714
            else:
                if (r < 0.62): 
                    f = str(q*5+3)+'/5' # 0.6
                else:
                    f = str(q*3+2)+'/3' # 0.6666...
        else:
            if (r < 0.80):
                if (r < 0.74):
                    f = str(q*7+5)+'/7' # 0.7142...
                else:
                    if (r < 0.77) :
                        f = str(q*4+3)+'/4' # 0.75
                    else:
                        f = str(q*9+7)+'/9' # 0.7777...
            else:
                if (r < 0.85):
                    if (r < 0.83):
                        f = str(q*5+4)+'/5' # 0.8
                    else:
                        f = str(q*6+5)+'/6' # 0.8333...
                else:
                    if (r < 0.87):
                        f = str(q*7+6)+'/7' # 0.8571
                    else:
                        if (r < 0.88):
                            f = str(q*8+7)+'/8' # 0.875
                        else:
                            if (r < 0.90):
                                f = str(q*9+8)+'/9' # 0.8888...
                            else:
                                if (r < 0.95):
                                    f = str(q*10+9)+'/10' # 0.9
                                else:
                                    f = str(q+1) # 1.0

    return sign+f

def unique(data):
    
    data_type = data.dtype
    item_size = data_type.itemsize
    data_size = data.shape[1]
    
    dtype = np.dtype((np.void, item_size*data_size))

    b = np.ascontiguousarray(data).view(dtype)
   
    u, ind, inv = np.unique(b, return_index=True, return_inverse=True)

    return u.view(data_type).reshape(-1, data_size), ind, inv

def evaluate(operator, coordinates, translate=True):

    operator = str(operator)

    x, y, z = coordinates

    if (not translate):
        ops = operator.split(',')
        #ops = [op.replace('/', '//') for op in ops]
        ops = [re.sub(r'\.', '', op) for op in ops]
        ops = [re.sub(r'\/', '', op) for op in ops]
        ops = [re.sub(r'[-+][\d]+', '', op) for op in ops]
        ops = [re.sub(r'[\d]', '', op) for op in ops]
        operator = ','.join(ops)

    return np.array(eval(operator))

def evaluate_mag(operator, moments):

    operator = str(operator)

    mx, my, mz = moments

    return np.array(eval(operator))

def evaluate_disp(operator, displacements):

    operator = str(operator)

    U11, U22, U33, U23, U13, U12 = displacements

    U = np.array([[U11,U12,U13],
                  [U12,U22,U23],
                  [U13,U23,U33]])

    W = np.zeros((3,3))

    W[:,0] = evaluate(operator, [1,0,0], translate=False)
    W[:,1] = evaluate(operator, [0,1,0], translate=False)
    W[:,2] = evaluate(operator, [0,0,1], translate=False)

    Up = np.dot(np.dot(W,U),W.T)

    return Up[0,0], Up[1,1], Up[2,2], Up[1,2], Up[0,2], Up[0,1]

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

        W_inv = np.linalg.inv(W)
        w_inv = -np.dot(W_inv, w)

        W_inv = W_inv.round().astype(int).astype(str)
        w_inv = [fraction(c) for c in w_inv]

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
            if (w_inv[i] != '0'):
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
    w = [fraction(c) for c in w]
            
    symop = u''

    for i in range(3):
        string = ''
        for j in range(3):
            if (W[i,j][0] == '-'):
                if (W[i,j] == '-1'):
                    string += '-'+uvw[j]
                else:
                    string += W[i,j]+'*'+uvw[j]
            elif (W[i,j] == '1'):
                if (len(string) == 0):
                    string += uvw[j]
                else:
                    string += '+'+uvw[j]
            elif (W[i,j] != '0'):
                if (len(string) == 0):
                    string += W[i,j]+'*'+uvw[j]
                else:
                    string += '+'+W[i,j]+'*'+uvw[j]
        if (w[i] != '0'):
            if (w[i][0] == '-'):
                string += w[i]
            else:
                string += '+'+w[i]
        symop += string
        if (i != 2):
            symop += ','

    return symop

def classification(symop):

    W = np.zeros((3,3))

    W[:,0] = evaluate(symop, [1,0,0], translate=False)
    W[:,1] = evaluate(symop, [0,1,0], translate=False)
    W[:,2] = evaluate(symop, [0,0,1], translate=False)

    W_det = np.linalg.det(W)
    W_tr = np.trace(W)
    
    if np.isclose(W_det, 1):
        if np.isclose(W_tr, 3):
            rotation, k = '1', 1
        elif np.isclose(W_tr, 2):
            rotation, k = '6', 6
        elif np.isclose(W_tr, 1):
            rotation, k = '4', 4
        elif np.isclose(W_tr, 0):
            rotation, k = '3', 3
        elif np.isclose(W_tr, -1):
            rotation, k = '2', 2
    elif np.isclose(W_det, -1):
        if np.isclose(W_tr, -3):
            rotation, k = '-1', 2
        elif np.isclose(W_tr, -2):
            rotation, k = '-6', 6
        elif np.isclose(W_tr, -1):
            rotation, k = '-4', 4
        elif np.isclose(W_tr, 0):
            rotation, k = '-3', 6
        elif np.isclose(W_tr, 1):
            rotation, k = 'm', 2

    symop_ord = symop

    for _ in range(1,k):
        symop_ord = binary(symop_ord, symop)
        
    wg = (1/k)*evaluate(symop_ord, [0,0,0], translate=True)

    return rotation, k, wg

def absence(symops, h, k, l):

    H = np.array([h,k,l])

    n = 1 if H.size == 3 else H.shape[1]

    if (n == 1): H = H.reshape(3,1)

    absent = np.full((len(symops),n), False)

    W = np.zeros((3,3))

    for i, symop in enumerate(symops):

        rotation, k, wg = classification(symop)

        W[:,0] = evaluate(symop, [1,0,0], translate=False)
        W[:,1] = evaluate(symop, [0,1,0], translate=False)
        W[:,2] = evaluate(symop, [0,0,1], translate=False)

        absent[i,:] = np.all(np.isclose(np.dot(H.T,W), H.T), axis=1) & \
                    ~ np.isclose(np.mod(np.dot(H.T,wg),1), 0)

    absent = absent.any(axis=0)

    if (H.shape[1] == 1):
        return absent[0]
    else:
        return absent

def site(symops, coordinates, A, tol=1e-1):

    u, v, w = coordinates

    W = np.zeros((3,3))

    metric = []
    operators = []

    U, V, W = np.meshgrid(np.arange(-1,2),
                          np.arange(-1,2),
                          np.arange(-1,2), indexing='ij')

    U = U.flatten()
    V = V.flatten()
    W = W.flatten()

    for i, symop in enumerate(symops):
        
        x, y, z = evaluate(symop, coordinates, translate=True)
        
        du, dv, dw = x-u, y-v, z-w

        if (du > 0.5): du -= 1
        if (dv > 0.5): dv -= 1
        if (dw > 0.5): dw -= 1

        if (du <= -0.5): du += 1
        if (dv <= -0.5): dv += 1
        if (dw <= -0.5): dw += 1

        nu, nv, nw = int(round(u-du)), int(round(v-dv)), int(round(w-dw))
        
        for iu, iv, iw in zip(U,V,W):

            cu, cv, cw = iu+nu, iv+nv, iw+nw

            tu, tv, tw = '', '', ''

            if (cu < 0): tu = '{}'.format(cu)
            if (cv < 0): tv = '{}'.format(cv)
            if (cw < 0): tw = '{}'.format(cw)

            if (cu > 0): tu = '+{}'.format(cu)
            if (cv > 0): tv = '+{}'.format(cv)
            if (cw > 0): tw = '+{}'.format(cw)

            trans_symop = binary('x{},y{},z{}'.format(tu,tv,tw), symop)
            
            x, y, z = evaluate(trans_symop, coordinates, translate=True)

            du, dv, dw = x-u, y-v, z-w

            dx, dy, dz = np.dot(A, [du,dv,dw])

            d = np.sqrt(dx**2+dy**2+dz**2)
                        
            if (d < tol):
                metric.append(d)
                operators.append(trans_symop)

    sort = np.argsort(metric)
    op = operators[sort[0]]
    
    W = np.zeros((3,3))
    w = np.zeros(3)

    W[:,0] = evaluate(op, [1,0,0], translate=False)
    W[:,1] = evaluate(op, [0,1,0], translate=False)
    W[:,2] = evaluate(op, [0,0,1], translate=False)

    w = evaluate(op, [0,0,0], translate=True)

    G = set({op})
    
    for i in range(len(operators)-1):

        op = operators[sort[i+1]]

        G.add(op)
        Gc = G.copy()
        
        for op_0 in Gc:
            for op_1 in Gc:
                if (op_0 != op_1):
                    symop = binary(op_0, op_1)

                    W[:,0] = evaluate(symop, [1,0,0], translate=False)
                    W[:,1] = evaluate(symop, [0,1,0], translate=False)
                    W[:,2] = evaluate(symop, [0,0,1], translate=False)

                    w = evaluate(symop, [0,0,0], translate=True)
                                        
                    if (np.allclose(W, np.eye(3)) and np.linalg.norm(w) > 0):
                        G.discard(op)
        
    T = np.zeros((3,3))
    t = np.zeros(3)

    rot = []
    for op in G:
        rotation, k, wg = classification(op)
        rot.append(rotation)
        
        W[:,0] = evaluate(op, [1,0,0], translate=False)
        W[:,1] = evaluate(op, [0,1,0], translate=False)
        W[:,2] = evaluate(op, [0,0,1], translate=False)
        
        w = evaluate(op, [0,0,0], translate=True)
        
        T += W
        t += w
                
    rot = np.array(rot)

    n_rot_6 = (rot == '6').sum()
    n_rot_4 = (rot == '4').sum()
    n_rot_3 = (rot == '3').sum()
    n_rot_2 = (rot == '2').sum()

    n_inv_1 = (rot == '-1').sum()
    n_inv_6 = (rot == '-6').sum()
    n_inv_4 = (rot == '-4').sum()
    n_inv_3 = (rot == '-3').sum()
    n_inv_2 = (rot == 'm').sum()

    nm = len(rot)
    mult = len(symops) // nm

    if (n_rot_3+n_inv_3 == 8):
        if (nm == 12):
            if (n_inv_1 == 0):
                pg ='23'
            else:
                pg = 'm-3'
        else:
            if (n_inv_1 == 0):
                if (n_rot_4 == 6):
                    pg = '432'
                else:
                    pg ='-43m'
            else:
                pg = 'm-3m'
    elif (n_rot_6+n_inv_6 == 2):
        if (nm == 6):
            if (n_inv_1 == 0):
                pg = '6'
            else:
                pg = '-6'
        else:
            if (n_inv_1 == 0):
                if (n_rot_6 == 2):
                    if (n_rot_2 == 7):
                        pg = '622'
                    else:
                        pg = '6mm'
                else:
                    pg = '6m-2'
            else:
                pg = '6/mmm'
    elif (n_rot_3+n_inv_3 == 2):
        if (nm == 3):
            if (n_inv_1 == 0):
                pg = '3'
            else:
                pg = '-3'
        else:
            if (n_inv_1 == 0):
                if (n_rot_2 == 3):
                    pg = '32'
                else:
                    pg = '3m'
            else:
                pg = '-3m'
    elif (n_rot_4+n_inv_4 == 2):
        if (nm == 4):
            if (n_inv_1 == 0):
                if (n_rot_4 == 2):
                    pg = '4'
                else:
                    pg = '-4'
            else:
                pg = '4/m'
        else:
            if (n_inv_1 == 0):
                if (n_rot_4 == 2):
                    if (n_rot_2 == 5):
                        pg = '422'
                    else:
                        pg = '4mm'
                else:
                    pg = '-4m2'
            else:
                pg = '4/mmm'
    elif (n_rot_2+n_inv_2 == 3):
        if (n_inv_1 == 0):
            if (n_rot_2 == 3):
                pg = '222'
            else:
                pg = 'mm2'
        else:
            pg = 'mmm'
    elif (n_rot_2+n_inv_2 == 1):
        if (n_inv_1 == 0):
            if (n_rot_2 == 1):
                pg = '2'
            else:
                pg = 'm'
        else:
            pg = '2/m'
    else:
        if (n_inv_1 == 0):
            pg = '1'
        else:
            pg = '-1'

    T /= nm
    t /= nm

    t = [fraction(c) for c in t]

    T = T.flatten()
    T = [fraction(c) for c in T]
    T = np.array(T).reshape(3,3).tolist()

    uvw = np.array(['x','y','z'])
    sppos = u''

    for i in range(3):
        string = ''
        for j in range(3):
            if (T[i][j] != '0'):
                if (T[i][j] == '1'):
                    if (len(string) > 0):
                        string += '+'
                    string += uvw[j]
                elif (T[i][j] == '-1'):
                    string += '-'+uvw[j]
                else:
                    if (len(string) > 0):
                        string += '+'
                    string += uvw[j]
        if (t[i] != '0'):
            if (t[i] == '-1'):
                string += '-'
            elif (len(string) > 0):
                string += '+'
            string += t[i]
        if (string == ''):
            string = '0'
        sppos += string
        if (i != 2):
            sppos += ','

    return pg, mult, sppos

def laue_id(symops):

    laue_sym = operators(invert=True)

    symop_id = [11,1]

    for c, sym in enumerate(list(laue_sym.keys())):
        if (np.array([symops[p] in laue_sym.get(sym) \
                      for p in range(symops.shape[0])]).all() and \
             len(laue_sym.get(sym)) == symops.shape[0]):

            symop_id = [c,len(laue_sym.get(sym))]

    return symop_id

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

    if invert:

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