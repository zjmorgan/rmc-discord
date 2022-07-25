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

def rotation_operator(val, col=0):

    if (col == 0):
        sym = 'x'
    elif (col == 1):
        sym = 'y'
    else:
        sym = 'z'

    if (val == 0):
        sign = ''
    elif (val > 0):
        if (col == 0):
            sign = ''
        else:
            sign = '+'
    else:
        sign = '-'

    val = abs(val)

    q, r = int(val // 1), val % 1

    if (r == 0):
        num, den = q, ''
    else:
        if (r < 0.47):
            if (r < 0.24):
                if (r < 0.16):
                    if (r < 0.12):
                        if (r < 0.11):
                            if (r < 0.09):
                                num, den = q, '' # 0.0
                            else:
                                num, den = q*10+1, '/10' # 0.1
                        else:
                            num, den = q*9+1, '/9' # 0.1111....
                    else:
                        if (r < 0.14):
                            num, den = q*8+1, '/8' # 0.125
                        else:
                            num, den = q*7+1, '/7' # 0.1428...
                else:
                    if (r < 0.19):
                        num, den = q*6+1, '/6' # 0.1666...
                    else:
                        if (r < 0.22):
                            num, den = q*5+1, '/5' # 0.2
                        else:
                            num, den = q*9+2, '/9' # 0.2222...
            else:
                if (r < 0.37):
                    if (r < 0.28):
                        num, den = q*4+1, '/4' # 0.25
                    else:
                        if (r < 0.29):
                            num, den = q*7+2, '/7' # 0.2857...
                        else:
                            if (r < 0.31):
                                num, den = q*10+3, '/10' # 0.3
                            else:
                                num, den = q*3+1, '/3' # 0.3333...
                else:
                    if (r < 0.42):
                        if (r < 0.40):
                            num, den = q*8+3, '/8' # 0.375
                        else:
                            num, den = q*5+2, '/5' # 0.4
                    else:
                        if (r < 0.44):
                            num, den = q*7+3, '/7' # 0.4285...
                        else:
                            num, den = q*9+4, '/9' # 0.4444...
        else:
            if (r < 0.71):
                if (r < 0.60):
                    if (r < 0.55):
                        num, den = q*2+1, '/2' # 0.5
                    else:
                        if (r < 0.57):
                            num, den = q*9+5, '/9' # 0.5555...
                        else:
                            num, den = q*7+4, '/7' # 0.5714
                else:
                    if (r < 0.62):
                        num, den = q*5+3, '/5' # 0.6
                    else:
                        if (r < 0.64):
                            num, den = q*8+5, '/8' # 0.625
                        else:
                            if (r < 0.68):
                                num, den = q*3+2, '/3' # 0.6666...
                            else:
                                num, den = q*10+7, '/10' # 0.7
            else:
                if (r < 0.80):
                    if (r < 0.74):
                        num, den = q*7+5, '/7' # 0.7142...
                    else:
                        if (r < 0.77) :
                            num, den = q*4+3, '/4' # 0.75
                        else:
                            num, den = q*9+7, '/9' # 0.7777...
                else:
                    if (r < 0.85):
                        if (r < 0.83):
                            num, den = q*5+4, '/5' # 0.8
                        else:
                            num, den = q*6+5, '/6' # 0.8333...
                    else:
                        if (r < 0.87):
                            num, den = q*7+6, '/7' # 0.8571
                        else:
                            if (r < 0.88):
                                num, den = q*8+7, '/8' # 0.875
                            else:
                                if (r < 0.90):
                                    num, den = q*9+8, '/9' # 0.8888...
                                else:
                                    if (r < 0.95):
                                        num, den = q*10+9, '/10' # 0.9
                                    else:
                                        num, den = q+1, '' # 1.0

    if (val == 0):
        w = ''
    elif (val == 1):
        w = sym
    elif (val % 1 == 0):
        w = str(num)+'*'+sym
    else:
        if (num == 1):
            w = sym+den
        else:
            w = str(num)+'*'+sym+den

    return sign+w

def translation_operator(val):

    if (val >= 0):
        sign = '+'
    else:
        sign = '-'

    val = abs(val)

    q, r = int(val // 1), val % 1

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
                    if (r < 0.29):
                        f = str(q*7+2)+'/7' # 0.2857...
                    else:
                        if (r < 0.31):
                            f = str(q*10+3)+'/10' # 0.3
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
                    if (r < 0.64):
                        f = str(q*8+5)+'/8' # 0.625
                    else:
                        if (r < 0.68):
                            f = str(q*3+2)+'/3' # 0.6666...
                        else:
                            f = str(q*10+7)+'/10' # 0.7
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

def evaluate(operators, coordinates, translate=True):

    code = evaluate_op(operators, translate=translate)

    return evaluate_code(code, coordinates)

def evaluate_op(operators, translate=True):

    operators = str([[op] for op in operators])

    if (not translate):
        operators = re.sub(r'\.', '', operators)
        operators = re.sub(r'\/', '', operators)
        operators = re.sub(r'[-+][\d]+', '', operators)
        operators = re.sub(r'[\d]', '', operators)
    operators = operators.replace("'","")

    return compile(operators, '<string>', 'eval')

def evaluate_code(code, coordinates):

    x, y, z = coordinates

    return np.array(eval(code))

def evaluate_mag(operators, moments):

    operators = str([[op] for op in operators])

    mx, my, mz = moments

    operators = operators.replace("'","")

    return np.array(eval(operators))

def evaluate_disp(operator, displacements):

    U11, U22, U33, U23, U13, U12 = displacements

    U = np.array([[U11,U12,U13],
                  [U12,U22,U23],
                  [U13,U23,U33]])

    code = evaluate_op(operator, translate=False)

    W_0 = evaluate_code(code, [1,0,0])
    W_1 = evaluate_code(code, [0,1,0])
    W_2 = evaluate_code(code, [0,0,1])

    W = np.hstack((W_0,W_1,W_2)).T.reshape(3,3).T

    Up = np.dot(np.dot(W,U),W.T)

    return Up[0,0], Up[1,1], Up[2,2], Up[1,2], Up[0,2], Up[0,1]

def reverse(symops):

    n = len(symops)

    w = evaluate(symops, [0,0,0], translate=True)

    w = w.reshape(n,3)

    code = evaluate_op(symops, translate=False)

    W_0 = evaluate_code(code, [1,0,0])
    W_1 = evaluate_code(code, [0,1,0])
    W_2 = evaluate_code(code, [0,0,1])

    W = np.hstack((W_0,W_1,W_2)).T.reshape(3,3,n).T

    W_inv = np.linalg.inv(W).round()

    w_inv = -np.einsum('ijk,ik->ij', W_inv, w)

    W_inv = np.array([rotation_operator(c,col=i%3) for i, c \
                      in enumerate(W_inv.flatten())])

    w_inv = np.array([translation_operator(c) for c in w_inv.flatten()])

    W_inv = W_inv.reshape(n,3,3)
    w_inv = w_inv.reshape(n,3)

    rev_symops = []
    for i in range(n):
        rev_symop = [u''.join(W_inv[i,0,:])+w_inv[i,0],
                     u''.join(W_inv[i,1,:])+w_inv[i,1],
                     u''.join(W_inv[i,2,:])+w_inv[i,2]]

        rev_symop = [op.lstrip('+') for op in rev_symop]
        rev_symop = [op.rstrip('0') for op in rev_symop]
        rev_symop = [op.rstrip('+') for op in rev_symop]

        rev_symop = ','.join(rev_symop)
        rev_symops.append(rev_symop)

    return rev_symops

def inverse(symops):

    n = len(symops)

    code = evaluate_op(symops, translate=False)

    W_0 = evaluate_code(code, [1,0,0])
    W_1 = evaluate_code(code, [0,1,0])
    W_2 = evaluate_code(code, [0,0,1])

    W = np.hstack((W_0,W_1,W_2)).T.reshape(3,3,n).T

    W_inv = np.linalg.inv(W).round()

    W_inv = np.array([rotation_operator(c,col=(i//3)%3) for i, c \
                      in enumerate(W_inv.flatten())])

    W_inv = W_inv.reshape(n,3,3)

    inv_symops = []
    for i in range(n):
        inv_symop = [u''.join(W_inv[i,:,0]),
                     u''.join(W_inv[i,:,1]),
                     u''.join(W_inv[i,:,2])]

        inv_symop = [op.lstrip('+') for op in inv_symop]
        inv_symop = [op.rstrip('0') for op in inv_symop]
        inv_symop = [op.rstrip('+') for op in inv_symop]

        inv_symop = ','.join(inv_symop)
        inv_symops.append(inv_symop)

    return inv_symops

def binary(symop0, symop1):

    n0, n1 = len(symop0), len(symop1)

    w0 = evaluate(symop0, [0,0,0], translate=True)
    w1 = evaluate(symop1, [0,0,0], translate=True)

    w0 = w0.reshape(n0,3)
    w1 = w1.reshape(n1,3)

    code0 = evaluate_op(symop0, translate=False)
    code1 = evaluate_op(symop1, translate=False)

    W0_0 = evaluate_code(code0, [1,0,0])
    W0_1 = evaluate_code(code0, [0,1,0])
    W0_2 = evaluate_code(code0, [0,0,1])

    W1_0 = evaluate_code(code1, [1,0,0])
    W1_1 = evaluate_code(code1, [0,1,0])
    W1_2 = evaluate_code(code1, [0,0,1])

    W0 = np.hstack((W0_0,W0_1,W0_2)).T.reshape(3,3,n0).T
    W1 = np.hstack((W1_0,W1_1,W1_2)).T.reshape(3,3,n1).T

    W = np.einsum('ijk,ikl->ijl', W0, W1).round()
    w = np.einsum('ijk,ik->ij', W0, w1)+w0

    W = np.array([rotation_operator(c,col=i%3) \
                  for i, c in enumerate(W.flatten())])
    w = np.array([translation_operator(c) for c in w.flatten()])

    W = W.reshape(n0,3,3)
    w = w.reshape(n0,3)

    symops = []
    for i in range(n0):
        symop = [u''.join(W[i,0,:])+w[i,0],
                 u''.join(W[i,1,:])+w[i,1],
                 u''.join(W[i,2,:])+w[i,2]]

        symop = [op.lstrip('+') for op in symop]
        symop = [op.rstrip('0') for op in symop]
        symop = [op.rstrip('+') for op in symop]

        symop = ','.join(symop)
        symops.append(symop)

    return symops

def binary_mag(symop0, symop1):

    n0, n1 = len(symop0), len(symop1)

    W0_0 = evaluate_mag(symop0, [1,0,0])
    W0_1 = evaluate_mag(symop0, [0,1,0])
    W0_2 = evaluate_mag(symop0, [0,0,1])

    W1_0 = evaluate_mag(symop1, [1,0,0])
    W1_1 = evaluate_mag(symop1, [0,1,0])
    W1_2 = evaluate_mag(symop1, [0,0,1])

    W0 = np.hstack((W0_0,W0_1,W0_2)).T.reshape(3,3,n0).T
    W1 = np.hstack((W1_0,W1_1,W1_2)).T.reshape(3,3,n1).T

    W = np.einsum('ijk,ikl->ijl', W0, W1).round()

    W = np.array([rotation_operator(c,col=i%3) \
                  for i, c in enumerate(W.flatten())])

    W = W.reshape(n0,3,3)

    symops = []
    for i in range(n0):
        symop = [u''.join(W[i,0,:]),
                 u''.join(W[i,1,:]),
                 u''.join(W[i,2,:])]

        symop = [op.lstrip('+') for op in symop]
        symop = [op.rstrip('0') for op in symop]
        symop = [op.rstrip('+') for op in symop]
        symop = ['0' if op == '' else op for op in symop]
        symop = [op.replace('x','mx') for op in symop]
        symop = [op.replace('y','my') for op in symop]
        symop = [op.replace('z','mz') for op in symop]

        symop = ','.join(symop)
        symops.append(symop)

    return symops

def classification(symops):

    n = len(symops)

    code = evaluate_op(symops, translate=False)

    W_0 = evaluate_code(code, [1,0,0])
    W_1 = evaluate_code(code, [0,1,0])
    W_2 = evaluate_code(code, [0,0,1])

    W = np.hstack((W_0,W_1,W_2)).T.reshape(3,3,n).T

    W_det = np.linalg.det(W)
    W_tr = np.trace(W, axis1=1, axis2=2)

    w = evaluate(symops, [0,0,0], translate=True)

    w_symop_ord = np.zeros((n,3))

    rotations, ks = [], []

    for i, symop in enumerate(symops):

        if np.isclose(W_det[i], 1):
            if np.isclose(W_tr[i], 3):
                rotation, k = '1', 1
            elif np.isclose(W_tr[i], 2):
                rotation, k = '6', 6
            elif np.isclose(W_tr[i], 1):
                rotation, k = '4', 4
            elif np.isclose(W_tr[i], 0):
                rotation, k = '3', 3
            elif np.isclose(W_tr[i], -1):
                rotation, k = '2', 2
        elif np.isclose(W_det[i], -1):
            if np.isclose(W_tr[i], -3):
                rotation, k = '-1', 2
            elif np.isclose(W_tr[i], -2):
                rotation, k = '-6', 6
            elif np.isclose(W_tr[i], -1):
                rotation, k = '-4', 4
            elif np.isclose(W_tr[i], 0):
                rotation, k = '-3', 6
            elif np.isclose(W_tr[i], 1):
                rotation, k = 'm', 2

        rotations.append(rotation)
        ks.append(k)

        W0, w0 = W[i,:,:].copy(), w[i,:].copy()
        W1, w1 = W[i,:,:].copy(), w[i,:].copy()

        for _ in range(1,k):
            W1 = np.dot(W0,W1)
            w1 = np.dot(W0,w1)+w0

        w_symop_ord[i,:] = w1

    k_inv = 1/np.array(ks)

    wgs = k_inv[:,np.newaxis]*w_symop_ord

    return rotations, ks, wgs.tolist()

def absence(symops, h, k, l):

    n = len(symops)

    H = np.array([h,k,l])

    m = 1 if H.size == 3 else H.shape[1]

    if (m == 1): H = H.reshape(3,1)

    absent = np.full((len(symops),m), False)

    W_0 = evaluate(symops, [1,0,0], translate=False)
    W_1 = evaluate(symops, [0,1,0], translate=False)
    W_2 = evaluate(symops, [0,0,1], translate=False)

    W = np.hstack((W_0,W_1,W_2)).T.reshape(3,3,n).T

    rotation, k, wg = classification(symops)

    for i in range(n):

        absent[i,:] = np.all(np.isclose(np.dot(H.T,W[i]), H.T), axis=1) & \
                    ~ np.isclose(np.mod(np.dot(H.T,wg[i]),1), 0)

    absent = absent.any(axis=0)

    if (m == 1):
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

    Ws, ws = [], []

    for symop in symops:

        xyz = evaluate([symop], coordinates, translate=True)

        x, y, z = np.array(xyz).flatten()

        du, dv, dw = x-u, y-v, z-w

        if (du > 0.5): du -= 1
        if (dv > 0.5): dv -= 1
        if (dw > 0.5): dw -= 1

        if (du <= -0.5): du += 1
        if (dv <= -0.5): dv += 1
        if (dw <= -0.5): dw += 1

        nu, nv, nw = int(round(u-du)), int(round(v-dv)), int(round(w-dw))

        w0 = np.array([nu,nv,nw])

        w1 = evaluate([symop], [0,0,0], translate=True).flatten()

        code = evaluate_op([symop], translate=False)

        W1_0 = evaluate_code(code, [1,0,0])
        W1_1 = evaluate_code(code, [0,1,0])
        W1_2 = evaluate_code(code, [0,0,1])

        W1 = np.hstack((W1_0,W1_1,W1_2)).T.reshape(3,3).T

        up, vp, wp = np.dot(W1, [u,v,w])+w1+w0

        up += U
        vp += V
        wp += W

        du, dv, dw = up-u, vp-v, wp-w

        dx, dy, dz = np.dot(A, [du,dv,dw])

        dist = np.sqrt(dx**2+dy**2+dz**2)

        mask = dist < tol

        Wo = np.array([rotation_operator(c,col=i%3) \
                      for i, c in enumerate(W1.flatten())])

        Wo = Wo.reshape(3,3)

        for (d, iu, iv, iw) in zip(dist[mask],U[mask],V[mask],W[mask]):

            w2 = w0+w1+np.array([iu,iv,iw])

            wo = np.array([translation_operator(c) for c in w2.flatten()])

            symop = [u''.join(Wo[0,:])+wo[0],
                     u''.join(Wo[1,:])+wo[1],
                     u''.join(Wo[2,:])+wo[2]]

            symop = [op.lstrip('+') for op in symop]
            symop = [op.rstrip('0') for op in symop]
            symop = [op.rstrip('+') for op in symop]

            trans_symop = symop

            operators += [','.join(trans_symop)]
            metric.append(d)

            Ws.append(W1)
            ws.append(w2)

    sort = np.argsort(metric)
    operators = [operators[i] for i in sort]

    op = operators[0]
    G = set({op})

    for i in range(1,len(operators)):

        op_0 = operators[i]

        code0 = evaluate_op([op_0], translate=False)

        W0_0 = evaluate_code(code0, [1,0,0])
        W0_1 = evaluate_code(code0, [0,1,0])
        W0_2 = evaluate_code(code0, [0,0,1])

        W0 = np.hstack((W0_0,W0_1,W0_2)).T.reshape(3,3).T

        w0 = evaluate([op_0], [0,0,0], translate=True)

        Gc = G.copy()
        G.add(op_0)

        for op_1 in Gc:
            if (op_0 != op_1):

                code1 = evaluate_op([op_1], translate=False)

                W1_0 = evaluate_code(code1, [1,0,0])
                W1_1 = evaluate_code(code1, [0,1,0])
                W1_2 = evaluate_code(code1, [0,0,1])

                W1 = np.hstack((W1_0,W1_1,W1_2)).T.reshape(3,3).T

                w1 = evaluate([op_1], [0,0,0], translate=True)

                W = np.dot(W0, W1)
                w = np.dot(W0, w1.flatten())+w0.flatten()

                if (np.allclose(W, np.eye(3)) and np.linalg.norm(w) > 0):
                    G.discard(op_0)

    n = 1

    T = np.zeros((n,3,3))
    t = np.zeros((n,3))

    rot = []
    for op in G:
        rotation, k, wg = classification([op])
        rot.append(rotation)

        code = evaluate_op([op], translate=False)

        W_0 = evaluate_code(code, [1,0,0])
        W_1 = evaluate_code(code, [0,1,0])
        W_2 = evaluate_code(code, [0,0,1])

        W = np.hstack((W_0,W_1,W_2)).T.reshape(3,3,n).T

        w = evaluate([op], [0,0,0], translate=True)

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

    T = np.array([rotation_operator(c, col=i%3) \
                  for i, c in enumerate(T.flatten())])
    t = np.array([translation_operator(c) for c in t.flatten()])

    T = T.reshape(n,3,3)
    t = t.reshape(n,3)

    sp_poss = []
    for i in range(n):
        sp_pos = [u''.join(T[i,0,:])+t[i,0],
                 u''.join(T[i,1,:])+t[i,1],
                 u''.join(T[i,2,:])+t[i,2]]

        sp_pos = [op.lstrip('+') for op in sp_pos]
        sp_pos = [op.rstrip('0') for op in sp_pos]
        sp_pos = [op.rstrip('+') for op in sp_pos]
        sp_pos = ['0' if op == '' else op for op in sp_pos]

        sp_pos = ','.join(sp_pos)
        sp_poss.append(sp_pos)

    return pg, mult, sp_pos

def laue_id(symops):

    n = len(symops)

    laue_sym = operators(invert=True)

    symop_id = [11,1]

    for c, sym in enumerate(list(laue_sym.keys())):

        all_symops = np.all([symops[p] in laue_sym.get(sym) for p in range(n)])
        len_symops = len(laue_sym.get(sym))

        if (all_symops and len_symops == n):

            symop_id = [c,len_symops]

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

        for symmetry in list(laue.keys()):

            laue[symmetry] = inverse(laue.get(symmetry))

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