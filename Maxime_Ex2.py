# Exercise 2 - Support Vector Machines with Radial Basis Functions

import matplotlib.pyplot as plt
import math

# data
Y = [1]*5 + [-1]*5
X1 = [1, 1, 2, 3, 3, 1, 3, 5, 5, 5]
X2 = [1, 3, 2, 1, 3, 5, 5, 1, 3, 5]
am = [0, 0, 1, 0, 0, 0, 1, 0, 1, 0]

# Question 1 ###################################################################

# In g, we only keep the terms where am is not null (3, 7 and 9).
# g(X, w) = y3*f(||X - X3||) + y7*f(||X - X7||) + y9*f(||X - X9||)
#         = f(||X - X3||) - f(||X - X7||) - f(||X - X9||)

def gaussian_f(x):
    return math.exp(- (x**2)/2)

def norm(x1, x2):
    return x1**2 + x2**2

def g(x1, x2, w):
    return gaussian_f(norm(x1 - X1[2], x2 - X2[2]))\
            - gaussian_f(norm(x1 - X1[6], x2 - X2[6]))\
            - gaussian_f(norm(x1 - X1[8], x2 - X2[8]))


# plot of the decision regions
# (bruteforce plot, sorry for that)
l_pos_x = []
l_pos_y = []
l_neg_x = []
l_neg_y = []
step = 30.0
for xbig in range(0, int(7*step)): # can't use range with float step => trick
    x = xbig/step
    for ybig in range(0, int(7*step)):
        y = ybig/step
        if g(x, y, 0) >= 0.0:
            l_pos_x.append(x)
            l_pos_y.append(y)
        else:
            l_neg_x.append(x)
            l_neg_y.append(y)

plt.scatter(l_pos_x, l_pos_y, color="green", marker='.', label="class 1")
plt.scatter(l_neg_x, l_neg_y, color="red", marker='.', label="class 2")

plt.scatter(X1[0:5], X2[0:5], color="black", marker="x", label="class 1", linewidths=1)
plt.scatter(X1[5:10], X2[5:10], color="black", marker="^", label="class 2", linewidths=1)
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend(loc='upper right')


# Question 2 ###################################################################

# if the training dataset is separable, then
# all ym*g(x) should be positive (correctly classifed)
separable = True
for i in range(10):
    if Y[i]*g(X1[i], X2[i], 0) < 0.0:
        separable = False
        break

print("The training dataset is" + (" " if separable else " not ") + "separable.")

plt.show()
