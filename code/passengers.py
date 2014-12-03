""" A set of functions to access and manipulate titanic passenger data


TODO: feature engineering
1. determine ethnicity from last name

TODO: feature normalization/fixing
1. remove ordinal values from categorical features
    - social class is ordinal, but title and embarked are not
2. normalize values to center around 0
    - active range for sigmoid function is also -sqrt(3) and sqrt(3)
"""

import numpy as np
import pandas
import math

bias = 1


def get_passengers(df):
    passengers = None
    for i, passenger in df.iterrows():
        # pclass
        if passenger['Pclass'] == 1:
            pclass = 0
        elif passenger['Pclass'] == 2:
            pclass = .5
        else:
            pclass = 1

        # sex
        if passenger['Sex'] == 'male':
            sex = 0
        else:
            sex = 1

        # embarked
        if passenger['Embarked'] == 'C':
            embarked = 0
        elif passenger['Embarked'] == 'S':
            embarked = .33
        elif passenger['Embarked'] == 'Q':
            embarked = .66
        else:
            embarked = 1

        # age - decreased general prediction by 2%
        if math.isnan(float(passenger['Age'])):
            age = .5 # fill-in...need to calculate median or something
        else:
            age = float(passenger['Age']) / 100 # normalize between 0 & 1

        # name (title)
        if 'Mr.' in passenger['Name']:
            name = 0
        elif 'Mrs.' in passenger['Name']:
            name = .25
        elif 'Master.' in passenger['Name']:
            name = .50
        elif 'Miss.' in passenger['Name']:
            name = .75
        else:
            name = 1

        try:
            if i == 0:
                passengers = np.matrix([bias, pclass, sex, embarked, age, name, passenger['Survived'], passenger['PassengerId']])
            else:
                passengers = np.vstack([passengers, [bias, pclass, sex, embarked, age, name, passenger['Survived'], passenger['PassengerId']]])
        except: # for test output
            if i == 0:
                passengers = np.matrix([bias, pclass, sex, embarked, age, name, passenger['PassengerId']])
            else:
                passengers = np.vstack([passengers, [bias, pclass, sex, embarked, age, name, passenger['PassengerId']]])
    return passengers
