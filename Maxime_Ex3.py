#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Exercise 3 - Artificial Neural Networks

import math
from numpy import mean

# Weights
# I made functions so that the indices are the same as
# in the subject without -1 everywhere in the code
_W = [[[1, 0], [-1, 0], [0, 1]], [[1, -1, 0]]]
_b = [[-1, 0, 1], [1]]

def W(j, i, l):
    """ returns w_ij^l """
    return _W[l-1][j-1][i-1]

def set_W(j, i, l, x):
    _W[l-1][j-1][i-1] = x

def b(j, l):
    """ returns b_j^l """
    return _b[l-1][j-1]

def set_b(j, l, x):
    _b[l-1][j-1] = x

# Training data
X = [(1, 0), (1, 1), (0,1), (0, 0)]
Y = [1, 0, 1, 0]

# sigmoid
def f(z):
    return 1/(1 + math.exp(-z))

# Question a ##################################################################
sample1 = True # I print only the values for the first sample
for (x1,x2),y in zip(X,Y):

    a1_1 = f(W(1,1,l=1)*x1 + W(1,2,l=1)*x2 + b(1,l=1))
    a2_1 = f(W(2,1,l=1)*x1 + W(2,2,l=1)*x2 + b(2,l=1))
    a3_1 = f(W(3,1,l=1)*x1 + W(3,2,l=1)*x2 + b(3,l=1))
    a1_2 = f(a1_1*W(1,1,l=2) + a2_1*W(1,2,l=2) + a3_1*W(1,3,l=2) + b(1,l=2))

    if (sample1):
        print("\n=== Question a : Feed forward (sample 1) ===")
        print("Layer 1 :")
        print("  a1 = ", a1_1)
        print("  a2 = ", a2_1)
        print("  a3 = ", a3_1)
        print("Layer 2 :")
        print("  a1 = ", a1_2)

# Question b ##################################################################
    d1_2 = a1_2 * (1 - a1_2) * (a1_2 - y)
    d1_1 = (a1_1) * (1 - a1_1) * W(1,1,l=2) * d1_2
    d2_1 = (a2_1) * (1 - a2_1) * W(1,2,l=2) * d1_2
    d3_1 = (a3_1) * (1 - a3_1) * W(1,3,l=2) * d1_2

    if (sample1):
        print("\n=== Question b : Back propagation (sample 1) ===")
        print("Layer 2 :")
        print("  d1 = ", d1_2)
        print("Layer 1 :")
        print("  d1 = ", d1_1)
        print("  d2 = ", d2_1)
        print("  d3 = ", d3_1)

# Question c ##################################################################
    dw11_2 = a1_1 * d1_2
    dw12_2 = a2_1 * d1_2
    dw13_2 = a3_1 * d1_2
    db1_2 = d1_2

    dw11_1 = x1 * d1_1
    dw21_1 = x1 * d2_1
    dw31_1 = x1 * d3_1
    dw12_1 = x2 * d1_1
    dw22_1 = x2 * d2_1
    dw32_1 = x2 * d3_1
    db1_1 = d1_1
    db2_1 = d2_1
    db3_1 = d3_1

    if (sample1):
        print("\n=== Question c : Weight corrections (sample 1) ===")
        print("Layer 2 :")
        print("  dw11 = ", dw11_2)
        print("  dw12 = ", dw12_2)
        print("  dw13 = ", dw13_2)
        print("  db1 = ", db1_2)
        print("Layer 1 :")
        print("  dw11 = ", dw11_1)
        print("  dw21 = ", dw21_1)
        print("  dw31 = ", dw31_1)
        print("  dw12 = ", dw12_1)
        print("  dw22 = ", dw22_1)
        print("  dw32 = ", dw32_1)
        print("  db1 = ", db1_1)
        print("  db2 = ", db2_1)
        print("  db3 = ", db3_1)

    sample1 = False

# Question d ##################################################################

    eta = 0.5 # learning rate
    set_W(1,1,1, W(1,1,1) - eta*dw11_1)
    set_W(2,1,1, W(2,1,1) - eta*dw21_1)
    set_W(3,1,1, W(3,1,1) - eta*dw31_1)
    set_W(1,2,1, W(1,2,1) - eta*dw12_1)
    set_W(2,2,1, W(2,2,1) - eta*dw22_1)
    set_W(3,2,1, W(3,2,1) - eta*dw32_1)
    set_b(1,1, b(1,1) - eta*db1_1)
    set_b(2,1, b(2,1) - eta*db2_1)
    set_b(3,1, b(3,1) - eta*db3_1)

    set_W(1,1,2, W(1,1,2) - eta*dw11_2)
    set_W(1,2,2, W(1,2,2) - eta*dw12_2)
    set_W(1,3,2, W(1,3,2) - eta*dw13_2)
    set_b(1,2, b(1,2) - eta*db1_2)


print("\n=== Question d : Resulting parameters (1st method) ===")
print("Layer 1 :")
print("w11 = ", W(1,1,1))
print("w21 = ", W(2,1,1))
print("w31 = ", W(3,1,1))
print("w12 = ", W(1,2,1))
print("w22 = ", W(2,2,1))
print("w32 = ", W(3,2,1))
print("b1 = ", b(1,1))
print("b2 = ", b(2,1))
print("b3 = ", b(3,1))
print("Layer 2 :")
print("w11 = ", W(1,1,2))
print("w12 = ", W(1,2,2))
print("w13 = ", W(1,3,2))
print("b1 = ", b(1,2))
    
# Question e ##################################################################

# reinitialize the weights, and apply the second method
_W = [[[1, 0], [-1, 0], [0, 1]], [[1, -1, 0]]]
_b = [[-1, 0, 1], [1]]

i = 0
dw11_2, dw12_2, dw13_2, db1_2, dw11_1, dw21_1, dw31_1, dw12_1, dw22_1, dw32_1,\
db1_1, db2_1, db3_1 = [list() for i in range(13)]

for (x1,x2),y in zip(X,Y):

    # units outputs
    a1_1 = f(W(1,1,l=1)*x1 + W(1,2,l=1)*x2 + b(1,l=1))
    a2_1 = f(W(2,1,l=1)*x1 + W(2,2,l=1)*x2 + b(2,l=1))
    a3_1 = f(W(3,1,l=1)*x1 + W(3,2,l=1)*x2 + b(3,l=1))
    a1_2 = f(a1_1*W(1,1,l=2) + a2_1*W(1,2,l=2) + a3_1*W(1,3,l=2) + b(1,l=2))
    
    # deltas of back propagation
    d1_2 = a1_2 * (1 - a1_2) * (a1_2 - y)
    d1_1 = (a1_1) * (1 - a1_1) * W(1,1,l=2) * d1_2
    d2_1 = (a2_1) * (1 - a2_1) * W(1,2,l=2) * d1_2
    d3_1 = (a3_1) * (1 - a3_1) * W(1,3,l=2) * d1_2
    
    # corrections
    dw11_2.append(a1_1 * d1_2)
    dw12_2.append(a2_1 * d1_2)
    dw13_2.append(a3_1 * d1_2)
    db1_2.append(d1_2)

    dw11_1.append(x1 * d1_1)
    dw21_1.append(x1 * d2_1)
    dw31_1.append(x1 * d3_1)
    dw12_1.append(x2 * d1_1)
    dw22_1.append(x2 * d2_1)
    dw32_1.append(x2 * d3_1)
    db1_1.append(d1_1)
    db2_1.append(d2_1)
    db3_1.append(d3_1)

    i += 1

eta = 0.5 # learning rate
set_W(1,1,1, W(1,1,1) - eta*mean(dw11_1))
set_W(2,1,1, W(2,1,1) - eta*mean(dw21_1))
set_W(3,1,1, W(3,1,1) - eta*mean(dw31_1))
set_W(1,2,1, W(1,2,1) - eta*mean(dw12_1))
set_W(2,2,1, W(2,2,1) - eta*mean(dw22_1))
set_W(3,2,1, W(3,2,1) - eta*mean(dw32_1))
set_b(1,1, b(1,1) - eta*mean(db1_1))
set_b(2,1, b(2,1) - eta*mean(db2_1))
set_b(3,1, b(3,1) - eta*mean(db3_1))

set_W(1,1,2, W(1,1,2) - eta*mean(dw11_2))
set_W(1,2,2, W(1,2,2) - eta*mean(dw12_2))
set_W(1,3,2, W(1,3,2) - eta*mean(dw13_2))
set_b(1,2, b(1,2) - eta*mean(db1_2))


print("\n=== Question e : Resulting parameters (2nd method) ===")
print("Layer 1 :")
print("w11 = ", W(1,1,1))
print("w21 = ", W(2,1,1))
print("w31 = ", W(3,1,1))
print("w12 = ", W(1,2,1))
print("w22 = ", W(2,2,1))
print("w32 = ", W(3,2,1))
print("b1 = ", b(1,1))
print("b2 = ", b(2,1))
print("b3 = ", b(3,1))
print("Layer 2 :")
print("w11 = ", W(1,1,2))
print("w12 = ", W(1,2,2))
print("w13 = ", W(1,3,2))
print("b1 = ", b(1,2))
 

