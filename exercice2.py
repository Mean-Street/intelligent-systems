from math import exp

def g(x, y):
    return exp(x**2 + y**2) * (exp(2*x + 2*y - 4) - exp(3*x + 5*y - 17) - exp(5*x + 3*y - 17))

def main():
    training_data = [(1, 1, 1, 0),
                     (1, 1, 3, 0),
                     (1, 2, 2, 1),
                     (1, 3, 1, 0),
                     (1, 3, 3, 0),
                     (-1, 1, 5, 0),
                     (-1, 3, 5, 1),
                     (-1, 5, 1, 0),
                     (-1, 5, 3, 1),
                     (-1, 5, 5, 0)]
    separable = True;
    for datum in training_data:
        ym = datum0]
        x = datum[1]
        y = datum[2]
        if ym*g(x, y) <= 0:
            separable = false
            break
    print(separable)
          
main()
