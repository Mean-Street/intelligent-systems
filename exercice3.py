from math import exp
import numpy as np

def f(z):
    """
    The sigmoid function
    """
    return 1/(1+exp(-z))

def main():
    #fill Wji
    W = []
    W.append([])
    W1 = [[ 1, 0],
            [-1, 0],
            [ 0, 1]]
    W.append(W1)
    
    W2 = [[1, -1, 0]]
    W.append(W2)

    #fill bj
    b = []
    b.append([])
    b1 = [-1, 0, 1]
    b.append(b1)
    b2 = [1]
    b.append(b2)

    #The number of activation units in each layer
    N = (2, 3, 1)

    #the numbers of layers
    L = len(N)

    #The training sample
    X1 = [1, 0]
    X2 = [1, 1]
    X3 = [0, 1]
    X4 = [0, 0]
    X = [X1, X2, X3, X4]
    M = len(X)

    Y = [1, 0, 1, 0]

    averageDeltaW = []
    averageDeltaB = []

    #print the weight matrix for each layer
    for l in range(1, L):
        print("Weight matrix for layer", l)
        for j in range(0, N[l]):
                for i in range(0, N[l - 1]):
                    print("W[", l, "][", j, "][", i, "] = ", W[l][j][i])

    #print the number of units relative to each layer
    print("Number of units relative to each layer", N)

    #print the training sample
    print("The training sample with their indicator variable:")
    for m in range(0, M):
        print("X", m, " = ", X[m], "with Y", m, " = ", Y[m])

    print()

    for m in range(0, M):
        print("Computations for the sample", m + 1)
        # the outputs of units
        a = [X[m]]

        #the weighted input to jth unit of layer l
        z = []
        z.append([])

        #errors
        errors = []

        #corrections
        DeltaW = []
        DeltaB = []

        #learning rate
        learning_rate = 0.5
    
        #question a
        #we iterate through the layers
        for l in range(1, L):
            a.append([])
            z.append([])
            for j in range(0, N[l]):
                somme = 0
                i = 0
                for x in a[l - 1]:
                    somme = somme + W[l][j][i]*x
                    i = i + 1
                somme = somme + b[l][j]
                z[l].append(somme)
                a[l].append(f(z[l][j]))

        #question b and c
        #We work out the error for the output layer
        for l in range(0, L):
            errors.append([])
            DeltaW.append([])
            DeltaB.append([])
        for j in range(0, N[L - 1]):
            errors[L - 1].append(z[L - 1][j]*(1 - z[L - 1][j])*(a[L - 1][j] - Y[m]))
            DeltaW[L - 1].append([])
            DeltaB[L - 1].append(errors[L - 1][j])
            for i in range(0, N[L - 2]):
                DeltaW[L - 1][j].append(a[L - 2][i] * errors[L - 1][j])
                
        #the errors for the hidden layers
        for l in range(L - 2, -1, -1):
            for j in range(0, N[l]):
                somme = 0
                k = 0
                for x in errors[l + 1]:
                    somme = somme + W[l+1][k][j] * x
                    k = k + 1
                errors[l].append(a[l][j]*(1 - a[l][j])*somme)
    
        #work out the corrections
        for l in range(1, L):
            for j in range(0, N[l]):
                DeltaW[l].append([])
                DeltaB[l].append(errors[l][j])
                for i in range(0, N[l - 1]):
                    DeltaW[l][j].append(a[l - 1][i]*errors[l][j])
    
        """#apply the corrections for each sample
        for l in range(1, L):
            for j in range(0, N[l]):
                b[l][j] = b[l][j] - learning_rate * DeltaB[l][j] 
                i = 0
                for i in range(0, N[l - 1]):
                    W[l][j][i] = W[l][j][i] - learning_rate * DeltaW[l][j][i]
                    i = i + 1"""

        #update the average correction
        if m != 0:
            for l in  range(1, L):
                for j in range(0, N[l]):
                    averageDeltaB[l][j] = averageDeltaB[l][j] + DeltaB[l][j]
                    for i in range(0, N[l - 1]):
                        averageDeltaW[l][j][i] = averageDeltaW[l][j][i] + DeltaW[l][j][i]
        else:
            averageDeltaW = DeltaW
            averageDeltaB = DeltaB
                    
        #print the activation output for each unit of each layer
        for l in range(0, L):
            print("the weighted input and the  activation output for each unit of layer", l)
            for j in range(0, N[l]):
                print("a[", l, ",", j, "] = ", a[l][j])
                if l != 0:
                    print("z[", l, ",", j, "] = ", z[l][j])

        #print the errors for each layer
        for l in range(0, L):
            print("The errors for the layer", l)
            for j in range(0, N[l]):
                print("errors[", l, "][", j, "] = ", errors[l][j])

        #print the weight matrix for each layer
        for l in range(1, L):
            print("The weight matrix for the layer", l)
            for j in range(0, N[l]):
                for i in range(0, N[l - 1]):
                    print("DeltaW[", l, "][", j, "][", i, "] = ", DeltaW[l][j][i])

        #print the bias matrix for each layer
        for l in range(1, L):
            print("The bias matrix for the layer", l)
            for j in range(0, N[l]):
                print("DeltaB[", l, "][", j, "] = ", DeltaB[l][j])
        print()

    #work out the average of the correction factors
    for l in  range(1, L):
        for j in range(0, N[l]):
            averageDeltaB[l][j] = averageDeltaB[l][j] / M
            for i in range(0, N[l - 1]):
                averageDeltaW[l][j][i] = averageDeltaW[l][j][i] / M

    #apply the corrections using average of the correction factors
        for l in range(1, L):
            for j in range(0, N[l]):
                b[l][j] = b[l][j] - learning_rate * averageDeltaB[l][j] 
                i = 0
                for i in range(0, N[l - 1]):
                    W[l][j][i] = W[l][j][i] - learning_rate * averageDeltaW[l][j][i]
                    i = i + 1

    #print the weight matrix for each layer after applying the average correction
    for l in range(1, L):
        print("The weight matrix for the layer", l)
        for j in range(0, N[l]):
            for i in range(0, N[l - 1]):
                print("AverageDeltaW[", l, "][", j, "][", i, "] = ", averageDeltaW[l][j][i])

    #print the bias matrix for each layer after applying the average correction
    for l in range(1, L):
        print("The bias matrix for the layer", l)
        for j in range(0, N[l]):
            print("AverageDeltaB[", l, "][", j, "] = ", averageDeltaB[l][j])

    #print the weight matrix for each layer
    for l in range(1, L):
        print("Weight matrix for layer", l)
        for j in range(0, N[l]):
                print("b[", l, "][", j, "] = ", b[l][j])
                for i in range(0, N[l - 1]):
                    print("W[", l, "][", j, "][", i, "] = ", W[l][j][i])
main()  
