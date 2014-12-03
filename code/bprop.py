import numpy as np
import pandas
import passengers

learning_rate = 0.01
epochs = 45


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


def output_training_predictions(a, b, xm):
    training_predictions = open('../data/predictions/training_predictions.csv', 'w+')
    training_predictions.write('PassengerId,Survived\n')
    for passenger in xm:
        training_predictions.write('%s, %s\n' % (str(int(passenger[0,7])), str(int(predict(passenger, a, b)))))
    training_predictions.close()


def output_test_predictions(a, b):
    df = pandas.read_csv('../data/test.csv')
    xm = passengers.get_passengers(df)

    training_predictions = open('../data/predictions/test_predictions.csv', 'w+')
    training_predictions.write('PassengerId,Survived\n')
    for passenger in xm:
        training_predictions.write('%s, %s\n' % (str(int(passenger[0,6])), str(int(predict(passenger, a, b)))))
    training_predictions.close()


def main():
    """ Run Neural Network"""
    df = pandas.read_csv('../data/train.csv')
    xm = passengers.get_passengers(df)

    # initial weights
    a = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    b = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    print a
    print b

    for i in range(0,epochs):
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

            #print a_der
            #print b_der
            #print z

            # for use in calculating perdiction accuracy as model progresses
            if z < .5:
                z_prediction = 0.0
            else:
                z_prediction = 1.0
            if target == z_prediction:
                correct_predictions += 1
            total_predictions += 1

            E = 1.0/2.0 * (z-target)**2
            errors.append(E)

            if len(derivatives) == 0:
                derivatives.append(b_der[0])
                derivatives.append(b_der[1])
                derivatives.append(b_der[2])
                derivatives.append(b_der[3])
                derivatives.append(b_der[4])
                derivatives.append(b_der[5])

                derivatives.append(a_der[0])
                derivatives.append(a_der[1])
                derivatives.append(a_der[2])
                derivatives.append(a_der[3])
                derivatives.append(a_der[4])
                derivatives.append(a_der[5])
            else:
                derivatives[0] += b_der[0]
                derivatives[1] += b_der[1]
                derivatives[2] += b_der[2]
                derivatives[3] += b_der[3]
                derivatives[4] += b_der[4]
                derivatives[5] += b_der[5]

                derivatives[6] += a_der[0]
                derivatives[7] += a_der[1]
                derivatives[8] += a_der[2]
                derivatives[9] += a_der[3]
                derivatives[10] += a_der[4]
                derivatives[11] += a_der[5]

        # update
        b[0] += derivatives[0]*(-learning_rate)
        b[1] += derivatives[1]*(-learning_rate)
        b[2] += derivatives[2]*(-learning_rate)
        b[3] += derivatives[3]*(-learning_rate)
        b[4] += derivatives[4]*(-learning_rate)
        b[5] += derivatives[5]*(-learning_rate)

        a[0] += derivatives[6]*(-learning_rate)
        a[1] += derivatives[7]*(-learning_rate)
        a[2] += derivatives[8]*(-learning_rate)
        a[3] += derivatives[9]*(-learning_rate)
        a[4] += derivatives[10]*(-learning_rate)
        a[5] += derivatives[11]*(-learning_rate)

        print a
        print b
        print 'MSE = ' + str(sum(errors)/len(errors))
        print 'Prediction Rate:' +  str(float(correct_predictions)/float(total_predictions))
        print '\n'

    output_training_predictions(a, b, xm)
    output_test_predictions(a, b)

if __name__ == "__main__":
    main()
