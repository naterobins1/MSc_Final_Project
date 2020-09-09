"""

Define a function to add in the concentration values to my np.array

Written for python 3


"""

import numpy as np

def getdata_addconc(filename,conc):
    data = np.genfromtxt(filename)
    CV = np.zeros(shape=(len(data)+1, 3))
    CV[0][0] = conc

    for i in range(len(data)):
        CV[i+1] = data[i]

    return CV
