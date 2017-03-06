# Exercise 1 - Performance Evaluation Metrics

import matplotlib.pyplot as plt

# Question 1 ###################################################################

# Input data
h1 = [2, 4, 6, 8]
h2 = [8, 6, 4, 2]
h = [x1+x2 for x1,x2 in zip(h1,h2)]
N = len(h)

# Discriminant function g
# We define g as h1/h, g[x] contains the value of g(x+1)
g = [x1/x for x1,x in zip(h1,h)]
print("g(x) =", g)

# Decision function d
def compute_d(elt, bias):
    return 1 if elt + bias >= 0 else 0

# Question 2 ###################################################################

# g(x) + Bx = 0  =>  Bx = -g(x)
Bx = [-x for x in g]
print("Bx =", Bx)
print()

# Question 3 ###################################################################

TPR = [0]*N
FPR = [0]*N

# for q. 5
PP = [0]*N
F1 = [0]*N

# We compute the number of positive and negative entries
P = sum(h1)
N = sum(h2)

for i in range(len(Bx)):
    # We compute the decision function for the current bias
    d = [compute_d(elt, Bx[i]) for elt in g]

    # We compute the True/False Positive/Negative
    TP, FP, TN, FN = [0]*4
    for j in range(len(d)):
        # if x=j+1 estimated as class 1,
        # h1[j] are true positives and h2[j] are false positives
        if d[j] == 1: 
            TP += h1[j]
            FP += h2[j]
        # the 'else' is symmetrical
        else:
            TN += h2[j]
            FN += h1[j]

    # True Positive Rate
    TPR[i] = TP/P

    # False Positive Rate
    FPR[i] = FP/N

    # for q. 5
    PP[i] = TP/(TP + FP)
    F1[i] = 2*TP/(2*TP + FP + FN)

print("TPR =", TPR)
print("FPR =", FPR)
print()

# Question 4 ###################################################################

# plot of the ROC curve
plt.plot(FPR, TPR)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# Question 5 ###################################################################

# Precision
print("Precision :", PP)

# Recall
print("Recall :   ", TPR) # Recall = TPR

# We see that there's an inverse relationship between them, that's coherent
# The computation of F1 confirms that result, as it's higher for the middle
# values (the closest ones to the upper-left corner of the plot)
print("F1 :       ", F1)

