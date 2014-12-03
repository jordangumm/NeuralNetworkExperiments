import numpy as np
import pandas
import math


def get_input_matrix(data):
    """ Return matrix of input vectors.

    Initialize matrix.  Iteratively fill out matrix.  Return matrix.
    """
    data.readline() # skip headers
    v = data.readline()
    v = v.split(',')
    v[3] = v[3].replace(',', '')
    xm = np.matrix([1, float(v[2]), float(v[6]), float(v[7]), float(v[1])])

    for v in data.readlines():
        if ',,' not in v: # blank entries!!!
            v = v.split(',')
            xm = np.vstack((xm, [1, float(v[2]), float(v[6]), float(v[7]), float(v[1])]))

    return xm


def get_derivatives(a, b, x, target):
    a_der = [] # hidden layer weights
    b_der = [] # output weights

    u = a[0] + x[0,1]*a[1] + x[0,2]*a[2] + x[0,3]*a[3] + x[0,4]*a[4] + x[0,5]*a[5] # linear input (weighted sum)
    y = 1/(1+(np.exp(-u))) # sigmoid hidden layer output
    v = b[0] + b[1]*y + b[2]*y + b[3]*y + b[4]*y + b[5]*y # linear input from hidden layer (weighted sum)
    z = 1/(1+(np.exp(-v))) # sigmout final output
    z_final = z # for export

    p = (z-target)* (z * (1-z)) # error derivative with respect to the weights
    b_der.append(p) # b[0] = p or 1/(1+(np.exp(-v)))

    for i in range(1,6): # number of derivatives

        # hidden layer mapping linear input function to sigmoid function
        u = a[0] + x[0,i]*a[i]
        y = 1/(1+(np.exp(-u)))

        # output layer mapping linear hidden layer input function to final sigmoid function
        v = b[0] + b[i]*y
        z = 1/(1+(np.exp(-v)))

        q = (p * b[i]) * y * (1-y)

        if i == 1: # save bias weight along with x1 weight
            b_der.append(p * y)

            a_der.append(q)
            a_der.append(q * x[0,1])
        else:
            b_der.append(p * y)
            a_der.append(q * x[0,i])

    return [a_der, b_der, z]


def predict(x, a, b):
    u = a[0] + x[0,1]*a[1] + x[0,2]*a[2] + x[0,3]*a[3] + x[0,4]*a[4] + x[0,5]*a[5] # linear input (weighted sum)
    y = 1/(1+(np.exp(-u))) # sigmoid hidden layer output
    v = b[0] + b[1]*y + b[2]*y + b[3]*y + b[4]*y + b[5]*y # linear input from hidden layer (weighted sum)
    z = 1/(1+(np.exp(-v))) # sigmout final output

    if z < .5:
        return '0'
    else:
        return '1'

""" UNUSED AS OF NOW
def output_training_predictions(a, b, xm):
    training_predictions = open('../data/predictions/training_predictions.csv', 'w+')
    training_predictions.write('PassengerId,Survived\n')
    for passenger in xm:
        training_predictions.write('%s, %s\n' % (str(int(passenger[0,7])), str(int(predict(passenger, a, b)))))
    training_predictions.close()


def output_test_predictions(a, b):
    df = pandas.read_csv('data/test.csv')
    xm = get_passengers(df)

    training_predictions = open('../data/predictions/test_predictions.csv', 'w+')
    training_predictions.write('PassengerId,Survived\n')
    for passenger in xm:
        training_predictions.write('%s, %s\n' % (str(int(passenger[0,6])), str(int(predict(passenger, a, b)))))
    training_predictions.close()
"""


""" Main Loop

Initial weights were updated to match the best found in bprop.py
There is no actual learning being done in the feed-forward NN.

"""
def main():
    xm = get_passengers()

    # initial weights
    a = np.array([ 1.12722457, -0.52870122, 2.64153881, -0.63675977, 0.85424979, 0.49468403])
    b = np.array([-0.6518991, -0.29197426, 3.11292552, -1.12959643, -2.27068905, 0.84231814])

    errors = []
    derivatives = []
    correct_predictions = 0
    total_predictions = 0
    for x in xm:
        target = x[0,6]

        # set derivatives and z
        output = get_derivatives(a, b, x, target)
        a_der = output[0]
        b_der = output[1]
        z = output[2]

        # for use in calculating perdiction accuracy
        if z < .5:
            z_prediction = 0.0
        else:
            z_prediction = 1.0
        if target == z_prediction:
            correct_predictions += 1
        total_predictions += 1

        E = 1.0/2.0 * (z-target)**2
        errors.append(E)

    print a
    print b
    print 'MSE = ' + str(sum(errors)/len(errors))
    print 'Prediction Rate:' +  str(float(correct_predictions)/float(total_predictions))
    print '\n'

if __name__ == "__main__":
    main()
