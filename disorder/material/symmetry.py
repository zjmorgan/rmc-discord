#!/usr/bin/env python3

import re

import numpy as np

def miller(h, k, l, sym, op):
    """
    Transform Miller indices according to Laue symmetry operators.

    ===== ===== =========
    Index Laue  Operators
    ===== ===== =========
    0     m-3m  48
    1     m-3   24
    2     6/mmm 24
    3     6/m   12
    4     -3m   12
    5     -3    6
    6     4/mmm 16
    7     4/m   8
    8     mmm   8
    9     2/m   4
    10    -1    2
    11    None  1
    ===== ===== =========

    Parameters
    ----------
    h, k, l : 1d array
        Miller indices.
    sym : int
        Index corresponding to Laue symmetry class.
    op : int
        Index corresponding to Laue symmetry operator.

    """

    if (sym == 0):

        if (op == 0): return h,k,l
        elif (op == 1): return -h,-k,l
        elif (op == 2): return -h,k,-l
        elif (op == 3): return h,-k,-l
        elif (op == 4): return l,h,k
        elif (op == 5): return l,-h,-k
        elif (op == 6): return -l,-h,k
        elif (op == 7): return -l,h,-k
        elif (op == 8): return k,l,h
        elif (op == 9): return -k,l,-h
        elif (op == 10): return k,-l,-h
        elif (op == 11): return -k,-l,h
        elif (op == 12): return k,h,-l
        elif (op == 13): return -k,-h,-l
        elif (op == 14): return k,-h,l
        elif (op == 15): return -k,h,l
        elif (op == 16): return h,l,-k
        elif (op == 17): return -h,l,k
        elif (op == 18): return -h,-l,-k
        elif (op == 19): return h,-l,k
        elif (op == 20): return l,k,-h
        elif (op == 21): return l,-k,h
        elif (op == 22): return -l,k,h
        elif (op == 23): return -l,-k,-h
        elif (op == 24): return -h,-k,-l
        elif (op == 25): return h,k,-l
        elif (op == 26): return h,-k,l
        elif (op == 27): return -h,k,l
        elif (op == 28): return -l,-h,-k
        elif (op == 29): return -l,h,k
        elif (op == 30): return l,h,-k
        elif (op == 31): return l,-h,k
        elif (op == 32): return -k,-l,-h
        elif (op == 33): return k,-l,h
        elif (op == 34): return -k,l,h
        elif (op == 35): return k,l,-h
        elif (op == 36): return -k,-h,l
        elif (op == 37): return k,h,l
        elif (op == 38): return -k,h,-l
        elif (op == 39): return k,-h,-l
        elif (op == 40): return -h,-l,k
        elif (op == 41): return h,-l,-k
        elif (op == 42): return h,l,k
        elif (op == 43): return -h,l,-k
        elif (op == 44): return -l,-k,h
        elif (op == 45): return -l,k,-h
        elif (op == 46): return l,-k,-h
        else: return l,k,h

    elif (sym == 1):

        if (op == 0): return h,k,l
        elif (op == 1): return -h,-k,l
        elif (op == 2): return -h,k,-l
        elif (op == 3): return h,-k,-l
        elif (op == 4): return l,h,k
        elif (op == 5): return l,-h,-k
        elif (op == 6): return -l,-h,k
        elif (op == 7): return -l,h,-k
        elif (op == 8): return k,l,h
        elif (op == 9): return -k,l,-h
        elif (op == 10): return k,-l,-h
        elif (op == 11): return -k,-l,h
        elif (op == 12): return -h,-k,-l
        elif (op == 13): return h,k,-l
        elif (op == 14): return h,-k,l
        elif (op == 15): return -h,k,l
        elif (op == 16): return -l,-h,-k
        elif (op == 17): return -l,h,k
        elif (op == 18): return l,h,-k
        elif (op == 19): return l,-h,k
        elif (op == 20): return -k,-l,-h
        elif (op == 21): return k,-l,h
        elif (op == 22): return -k,l,h
        else: return k,l,-h

    elif (sym == 2):

        if (op == 0): return h,k,l
        elif (op == 1): return -h-k,h,l
        elif (op == 2): return k,-h-k,l
        elif (op == 3): return -h,-k,l
        elif (op == 4): return h+k,-h,l
        elif (op == 5): return -k,h+k,l
        elif (op == 6): return k,h,-l
        elif (op == 7): return h,-h-k,-l
        elif (op == 8): return -h-k,k,-l
        elif (op == 9): return -k,-h,-l
        elif (op == 10): return -h,h+k,-l
        elif (op == 11): return h+k,-k,-l
        elif (op == 12): return -h,-k,-l
        elif (op == 13): return h+k,-h,-l
        elif (op == 14): return -k,h+k,-l
        elif (op == 15): return h,k,-l
        elif (op == 16): return -h-k,h,-l
        elif (op == 17): return k,-h-k,-l
        elif (op == 18): return -k,-h,l
        elif (op == 19): return -h,h+k,l
        elif (op == 20): return h+k,-k,l
        elif (op == 21): return k,h,l
        elif (op == 22): return h,-h-k,l
        else: return -h-k,k,l

    elif (sym == 3):

        if (op == 0): return h,k,l
        elif (op == 1): return -h-k,h,l
        elif (op == 2): return k,-h-k,l
        elif (op == 3): return -h,-k,l
        elif (op == 4): return h+k,-h,l
        elif (op == 5): return -k,h+k,l
        elif (op == 6): return -h,-k,-l
        elif (op == 7): return h+k,-h,-l
        elif (op == 8): return -k,h+k,-l
        elif (op == 9): return h,k,-l
        elif (op == 10): return -h-k,h,-l
        else: return k,-h-k,-l

    elif (sym == 4):

        if (op == 0): return h,k,l
        elif (op == 1): return -h-k,h,l
        elif (op == 2): return k,-h-k,l
        elif (op == 3): return -k,-h,-l
        elif (op == 4): return -h,h+k,-l
        elif (op == 5): return h+k,-k,-l
        elif (op == 6): return -h,-k,-l
        elif (op == 7): return h+k,-h,-l
        elif (op == 8): return -k,h+k,-l
        elif (op == 9): return k,h,l
        elif (op == 10): return h,-h-k,l
        else: return -h-k,k,l

    elif (sym == 5):

        if (op == 0): return h,k,l
        elif (op == 1): return -h-k,h,l
        elif (op == 2): return k,-h-k,l
        elif (op == 3): return -h,-k,-l
        elif (op == 4): return h+k,-h,-l
        else: return -k,h+k,-l

    elif (sym == 6):

        if (op == 0): return h,k,l
        elif (op == 1): return -h,-k,l
        elif (op == 2): return -k,h,l
        elif (op == 3): return k,-h,l
        elif (op == 4): return -h,k,-l
        elif (op == 5): return h,-k,-l
        elif (op == 6): return k,h,-l
        elif (op == 7): return -k,-h,-l
        elif (op == 8): return -h,-k,-l
        elif (op == 9): return h,k,-l
        elif (op == 10): return k,-h,-l
        elif (op == 11): return -k,h,-l
        elif (op == 12): return h,-k,l
        elif (op == 13): return -h,k,l
        elif (op == 14): return -k,-h,l
        else: return k,h,l

    elif (sym == 7):

        if (op == 0): return h,k,l
        elif (op == 1): return -h,-k,l
        elif (op == 2): return -k,h,l
        elif (op == 3): return k,-h,l
        elif (op == 4): return -h,-k,-l
        elif (op == 5): return h,k,-l
        elif (op == 6): return k,-h,-l
        else: return -k,h,-l

    elif (sym == 8):

        if (op == 0): return h,k,l
        elif (op == 1): return -h,-k,l
        elif (op == 2): return -h,k,-l
        elif (op == 3): return h,-k,-l
        elif (op == 4): return -h,-k,-l
        elif (op == 5): return h,k,-l
        elif (op == 6): return h,-k,l
        else: return -h,k,l

    elif (sym == 9):

        if (op == 0): return h,k,l
        elif (op == 1): return -h,k,-l
        elif (op == 2): return -h,-k,-l
        else: return h,-k,l

    elif (sym == 10):

        if (op == 0): return h,k,l
        else: return -h,-k,-l

    else:

        return h,k,l

def rotation_operator(val, col):
    """
    Convert values to rotation operator.

    Parameters
    ----------
    val : float
        Value of rotation.
    col : int
        Column of symmetry operator (``0``, ``1``, or ``2``).

    Returns
    -------
    op : str
        Rotation operator.

    """

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
    """
    Convert values to translation operator.

    Parameters
    ----------
    val : float
        Value of translation.

    Returns
    -------
    op : str
        Translation operator.

    """

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
    """
    Unique values, their indices in the original array, and the indices of the
    unique array to reconstruct the original array

    Parameters
    ----------
    data : 1d-array
        Array with values of either string or integer type.

    Returns
    -------
    uni, 1d-array
        Unique values.
    ind : 1d-array, int
        Indices of unique values in the original.
    inv : 1d-array, int
        Indices of the unique array that reconstructs the original.

    """

    data_type = data.dtype
    item_size = data_type.itemsize
    data_size = data.shape[1]

    dtype = np.dtype((np.void, item_size*data_size))

    b = np.ascontiguousarray(data).view(dtype)

    u, ind, inv = np.unique(b, return_index=True, return_inverse=True)

    uni_size = u.shape[0]

    return u.view(data_type).reshape(uni_size, data_size), ind, inv

def evaluate(operators, coordinates, translate=True):
    """
    Evaluate symmetry operators.

    Parameters
    ----------
    operators : list, str
        Symmetry operators.
    coordinates : 3-list
        Coordiantes to transform.
    translate : bool, optional
        Apply translation to rotation operator. The default is ``True``.

    Returns
    -------
    transformed : list
        Transformed coordinates.

    """

    code = evaluate_op(operators, translate=translate)

    return evaluate_code(code, coordinates)

def evaluate_op(operators, translate=True):
    """
    Compile evaluation code for symmetry operators.

    Parameters
    ----------
    operators : list, str
        Symmetry operators.
    translate : bool, optional
        Apply translation to rotation operator. The default is ``True``.

    Returns
    -------
    code : object
        Code object that can be evaluated.

    """

    operators = str([[op] for op in operators])

    if not translate:
        operators = re.sub(r'\.', '', operators)
        operators = re.sub(r'\/', '', operators)
        operators = re.sub(r'[-+][\d]+', '', operators)
        operators = re.sub(r'[\d]', '', operators)
    operators = operators.replace("'","")

    return compile(operators, '<string>', 'eval')

def evaluate_code(code, coordinates):
    """
    Evaluate code object over coordinates.

    Parameters
    ----------
    code : object
        Code object that can be evaluated.
    coordinates : 3-list
        Coordiantes to transform.

    Returns
    -------
    values : 2d-array, float
        Evaluated coordinates. Last axis has size of 3.

    """

    x, y, z = coordinates

    return np.array(eval(code))

def generate_mag(operator, symmform, parity):
    """
    Generate magnetic symmetry operator.

    Parameters
    ----------
    operator : list, str
        Symmetry operator.
    symmform : 3-list
        Magnetic symmetry form.
    parity : 3-list
        Moment parity.

    Returns
    -------
    mag_symops : list
        Magnetic symmetry operator.

    """

    code = evaluate_op(operator, translate=False)

    W_0 = evaluate_code(code, [1,0,0])
    W_1 = evaluate_code(code, [0,1,0])
    W_2 = evaluate_code(code, [0,0,1])

    W = np.hstack((W_0,W_1,W_2)).T.reshape(3,3).T

    M = (parity*np.linalg.det(W)*W).astype(int)

    symmform = symmform.split(',')

    mag_ops = []
    for i in range(3):
        mag_op = []
        for j in range(3):
            if M[i,j] > 0:
                mag_op.append(symmform[j])
            elif M[i,j] < 0:
                mag_op.append('-'+symmform[j])
        mag_op = '+'.join(mag_op)
        mag_op = mag_op.replace('--','')
        mag_op = mag_op.replace('+-','-')
        mag_ops.append(mag_op)

    mag_ops = ','.join(mag_ops)

    return mag_ops

def evaluate_mag(operator, moments):
    """
    Evaluate magnetic symmetry operator.

    Parameters
    ----------
    operator : list, str
        Magnetic symmetry operator.
    moment : 3-list
        Moments to transform.

    Returns
    -------
    transformed : list
        Transformed moments.

    """

    operators = str([[op] for op in operator])

    mx, my, mz = moments

    operators = operators.replace("'","")

    return np.array(eval(operators))

def evaluate_disp(operator, displacements):
    """
    Evaluate atomic displacement symmetry operator.

    Parameters
    ----------
    operator : list, str
        Symmetry operator.
    displacement : 6-list
        Atomic displacement parameters to transform.

    Returns
    -------
    transformed : list
        Transformed atomic displacement parameters.

    """

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
    """
    Reverse symmetry operators. Includes translation.

    Parameters
    ----------
    symops : list, str
        Symmetry operators.

    Returns
    -------
    rev_symops : list, str
        Reverse operators.

    """

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
    """
    Inverse symmetry operators. Removes translation.

    Parameters
    ----------
    symops : list, str
        Symmetry operators.

    Returns
    -------
    inv_symops : list, str
        Inverse operators.

    """

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
    """
    Binary operation betwen symmetry operators.

    Parameters
    ----------
    symop0 : list, str
        Symmetry operators.
    symop1 : list, str
        Symmetry operators. Must contain either 1 operator or same number of
        operators as the first.

    Returns
    -------
    symops : list, str
        Binary operations of the second acting on the first operator list.

    """

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

def classification(symops):
    """
    Symmetry operator classification.

    ===== ==== =====
    Determinant +1
    ----------------
    Trace Type Order
    ===== ==== =====
    3      1    1
    2      6    6
    1      4    4
    0      3    3
    -1     2    2
    ===== ==== =====

    ===== ==== =====
    Determinant -1
    ----------------
    Trace Type Order
    ===== ==== =====
    -3    -1   1
    -2    -6   6
    -1    -4   4
    0     -3   3
    1     m    2
    ===== ==== =====

    Parameters
    ----------
    symops : list, str
        Symmetry operators.

    Returns
    -------
    rotation : list, str
        Rotation type.
    k : list, int
        Order of operation.
    wg : list, str
        Glide or screw vector.

    """

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

    rotation, k = [], []

    for i, symop in enumerate(symops):

        if np.isclose(W_det[i], 1):
            if np.isclose(W_tr[i], 3):
                rot, order = '1', 1
            elif np.isclose(W_tr[i], 2):
                rot, order = '6', 6
            elif np.isclose(W_tr[i], 1):
                rot, order = '4', 4
            elif np.isclose(W_tr[i], 0):
                rot, order = '3', 3
            elif np.isclose(W_tr[i], -1):
                rot, order = '2', 2
        elif np.isclose(W_det[i], -1):
            if np.isclose(W_tr[i], -3):
                rot, order = '-1', 2
            elif np.isclose(W_tr[i], -2):
                rot, order = '-6', 6
            elif np.isclose(W_tr[i], -1):
                rot, order = '-4', 4
            elif np.isclose(W_tr[i], 0):
                rot, order = '-3', 6
            elif np.isclose(W_tr[i], 1):
                rot, order = 'm', 2

        rotation.append(rot)
        k.append(order)

        W0, w0 = W[i,:,:].copy(), w[i,:].copy()
        W1, w1 = W[i,:,:].copy(), w[i,:].copy()

        for _ in range(1,order):
            W1 = np.dot(W0,W1)
            w1 = np.dot(W0,w1)+w0

        w_symop_ord[i,:] = w1

    k_inv = 1/np.array(k)

    wg = k_inv[:,np.newaxis]*w_symop_ord

    return rotation, k, wg.tolist()

def absence(symops, h, k, l):
    """
    Systematic absence of reflections.

    Parameters
    ----------
    symops : list, str
        Symmetry operators.
    h, k, l : int
        Miller indices.

    Returns
    -------
    absent : bool
         Reflection indication of sytematic extinction.

    """

    n = len(symops)

    H = np.array([h,k,l])

    m = 1 if H.size == 3 else H.shape[1]

    if m == 1:
        H = H.reshape(3,1)

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

    if m == 1:
        return absent[0]
    else:
        return absent

def site(symops, coordinates, A, tol=1e-1):
    """
    Site symmetry.

    Parameters
    ----------
    symops : 1d array, str
        Symmetry operators of space group.
    coordinates : 2d array
        Factional coordinates.
    A : 2d array, 3x3
        Crystal to Cartesian axis coodinate system transformation matrix.
    tol : int, optional
        Distance tolerance. The default is ``1e-1``.

    Returns
    -------
    pg : 1d array, str
        Point group.
    mult : 1d array, int
        Multiplicity.
    sp_pos : 1d array, str
        Special position.

    """

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

    sp_pos = []
    for i in range(n):
        pos = [u''.join(T[i,0,:])+t[i,0],
               u''.join(T[i,1,:])+t[i,1],
               u''.join(T[i,2,:])+t[i,2]]

        pos = [op.lstrip('+') for op in pos]
        pos = [op.rstrip('0') for op in pos]
        pos = [op.rstrip('+') for op in pos]
        pos = ['0' if op == '' else op for op in pos]

        pos = ','.join(pos)
        sp_pos.append(pos)

    return pg, mult, sp_pos

def laue_id(symops):
    """
    Laue symmetry identifier.

    Parameters
    ----------
    symops : 1d array, str
        Array of symmetry operators.

    Returns
    -------
    symop_id : 2-list, int
        Laue symmetry identifier. First number is identifies and second is
        number of symmetry operators.

    """

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
    """
    Laue symmetry class and operators.

    ===== =========
    Laue  Operators
    ===== =========
    m-3m  48
    m-3   24
    6/mmm 24
    6/m   12
    -3m   12
    -3    6
    4/mmm 16
    4/m   8
    mmm   8
    2/m   4
    -1    2
    None  1
    ===== =========

    Parameters
    ----------
    invert : bool, optional
        Invert the Laue symmetry operators for reciprocal space. Default is
        ``False`` for real space operators.

    Returns
    -------
    laue : dictioanry
        Symmetry operators with Laue class keys.

    """

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
    """
    Laue symmetry operators.

    ===== =========
    Laue  Operators
    ===== =========
    m-3m  48
    m-3   24
    6/mmm 24
    6/m   12
    -3m   12
    -3    6
    4/mmm 16
    4/m   8
    mmm   8
    2/m   4
    -1    2
    None  1
    ===== =========

    Parameters
    ----------
    symmetry : stry
         Laue symmetry class.  One of ``'-1'``, ``'2/m'``, ``'mmm'``,
         ``'4/m'``, ``'4/mmm'``,  ``'-3'``, ``'-3m'``, ``'6/m'``, ``'6/mmm'``,
         ``'m-3'``, or ``'m-3m'``.

    Returns
    -------
    ops : list, str
        Symmetry operators.

    """

    if symmetry == 'm-3m':

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

    elif symmetry == 'm-3':

        ops = [u'x,y,z',u'-x,-y,z',u'-x,y,-z',u'x,-y,-z',
               u'z,x,y',u'z,-x,-y',u'-z,-x,y',u'-z,x,-y',
               u'y,z,x',u'-y,z,-x',u'y,-z,-x',u'-y,-z,x',
               u'-x,-y,-z',u'x,y,-z',u'x,-y,z',u'-x,y,z',
               u'-z,-x,-y',u'-z,x,y',u'z,x,-y',u'z,-x,y',
               u'-y,-z,-x',u'y,-z,x',u'-y,z,x',u'y,z,-x']

    elif symmetry == '6/mmm':

        ops = [u'x,y,z',u'-y,x-y,z',u'-x+y,-x,z',u'-x,-y,z',
               u'y,-x+y,z',u'x-y,x,z',u'y,x,-z',u'x-y,-y,-z',
               u'-x,-x+y,-z',u'-y,-x,-z',u'-x+y,y,-z',u'x,x-y,-z',
               u'-x,-y,-z',u'y,-x+y,-z',u'x-y,x,-z',u'x,y,-z',
               u'-y,x-y,-z',u'-x+y,-x,-z',u'-y,-x,z',u'-x+y,y,z',
               u'x,x-y,z',u'y,x,z',u'x-y,-y,z',u'-x,-x+y,z']

    elif symmetry == '6/m':

        ops = [u'x,y,z',u'-y,x-y,z',u'-x+y,-x,z',u'-x,-y,z',
               u'y,-x+y,z',u'x-y,x,z',u'-x,-y,-z',u'y,-x+y,-z',
               u'x-y,x,-z',u'x,y,-z',u'-y,x-y,-z',u'-x+y,-x,-z']

    elif symmetry == '-3m':

        ops = [u'x,y,z',u'-y,x-y,z',u'-x+y,-x,z',u'-y,-x,-z',
               u'-x+y,y,-z',u'x,x-y,-z',u'-x,-y,-z',u'y,-x+y,-z',
               u'x-y,x,-z',u'y,x,z',u'x-y,-y,z',u'-x,-x+y,z']

    elif symmetry == '-3':

        ops = [u'x,y,z',u'-y,x-y,z',u'-x+y,-x,z',
               u'-x,-y,-z',u'y,-x+y,-z',u'x-y,x,-z']

    elif symmetry == '4/mmm':

        ops = [u'x,y,z',u'-x,-y,z',u'-y,x,z',u'y,-x,z',
               u'-x,y,-z',u'x,-y,-z',u'y,x,-z',u'-y,-x,-z',
               u'-x,-y,-z',u'x,y,-z',u'y,-x,-z',u'-y,x,-z',
               u'x,-y,z',u'-x,y,z',u'-y,-x,z',u'y,x,z']

    elif symmetry == '4/m':

        ops = [u'x,y,z',u'-x,-y,z',u'-y,x,z',u'y,-x,z',
               u'-x,-y,-z',u'x,y,-z',u'y,-x,-z',u'-y,x,-z']

    elif symmetry == 'mmm':

        ops = [u'x,y,z',u'-x,-y,z',u'-x,y,-z',u'x,-y,-z',
               u'-x,-y,-z',u'x,y,-z',u'x,-y,z',u'-x,y,z']

    elif symmetry == '2/m':

        ops = [u'x,y,z',u'-x,y,-z',u'-x,-y,-z',u'x,-y,z']

    elif symmetry == '-1':

        ops = [u'x,y,z',u'-x,-y,-z']

    else:

        ops = [u'x,y,z']

    return ops
