# -*- coding: utf-8 -*-
"""

This is some test code to load the data that I am interested in a have a fiddle around wih

"""

#IMPORTS
import matplotlib.pyplot as plt
import numpy as np

#the root directory to access different files
rootdir = "/Users/nathanrobins/Documents/MSc_Neurotech/MSc_Project/Data"

#specify which CV file to look at --> integrate this into a for loop at some point
in_vitro = "/In_vitro_basal_5-HT"
in_vivo = "/In-vivo"
names = ["/101418", "/102218", "/110518", "/110718", "/110818","/102418"]

electrode = "/electrode1"
conc = 100
conc_file = "/"+str(conc)+"nm"
CV_file = "/100_CV.txt"

#for i in range(len(names)):
#for loop to go through each folder in in_vitro

#open the file
file = rootdir+in_vitro+names[0]+electrode+conc_file+CV_file
CV = np.genfromtxt(file)

list = False #false I think is better

while list is True:
    #convert into a list
    CV = CV.tolist()


    print((CV))
    print(CV[0])

    #time = CV_l[:,0]
    #current = CV_l[:,2]
else:
    #here I am keeping CV as a np.array
    concentration = np.array([conc,0,0], ndmin=2)
    CV = np.concatenate((concentration,CV))

    time = CV[1:,0]
    current = CV[1:,2]

print(CV)
#plot this file just to double check
plt.plot(time,current)
#plt.show()