"""

A script full of different functions to perform integration


"""

#imports
import scipy.integrate as integrate
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from Integration.add_conc import getdata_addconc
import numpy as np

def some_int(CV):
    if isinstance(CV,list) is True:
        CV = np.array(CV)
    #find time & current
    time = CV[1:, 0]
    current = CV[1:, 2]
    current = current/max(current)

    peaks, _ = find_peaks(-current)
    index = peaks[0:2]

    #draw a line between the 2 troughs
    coeff = np.polyfit((time[index[0]],time[index[1]]),(current[index[0]],current[index[1]]),1) #line with degree 1 is a straight line
    line = (coeff[0] * time) + coeff[1] #in the form mx + c

    #perform some simple integration to obtain a charge
    curve_a = integrate.trapz(current[index[0]:index[1]],time[index[0]:index[1]])
    line_a = integrate.trapz(line[index[0]:index[1]],time[index[0]:index[1]])

    charge = curve_a - line_a

    #visualisation
    plt.plot(time, current,'k')
    plt.plot(time[index[0]:index[1]],line[index[0]:index[1]],'r--')
    for i in range(len(peaks)): #plot all of the peaks
        plt.axvline(x=time[peaks[i]])

    #plt.title('Concentration = 25nm')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalised Charge')
    plt.grid()
    plt.show()
    plt.clf()
    return charge

def int_2(CV,visualisation = bool):
    #use this function as some_int(CV) does not work as find_peaks cannot find the small peak in the 10nm concs

    #find time & current
    time = CV[1:, 0]
    current = CV[1:, 2]

    #use find peaks to find the first peak & then use derivatives to find the second peak?
    #find_peaks seems to be pretty consistent for finding the first peak but not the second
    peaks, _ = find_peaks(-current)
    f_peak_index = peaks[0]

    #now find the second peak
    #calculate the derivative of the plot: change in y / change in x
    #change in x is just from time t to time t+1
    #with n data pts, the derivative will have n-1 data pts
    der = np.zeros(len(time)-1)
    for i in range(len(der)):
        dy = current[i+1]-current[i]
        dx = time[i+1] - time[i]
        der[i]=(dy/dx)*0.0001


    #find peaks/troughs of the derivative function
    der_peaks,_ = find_peaks(-der)

    #have a look at the double derivative
    derder = np.zeros(len(der) - 1)
    for i in range(len(derder)):
        dy = der[i + 1] - der[i]
        dx = time[i + 1] - time[i]
        derder[i] = (dy / dx) * 0.0001

    derder_peaks,_ = find_peaks(-derder)

    derderder = np.zeros(len(derder) - 1)
    for i in range(len(derderder)):
        dy = derder[i + 1] - derder[i]
        dx = time[i + 1] - time[i]
        derderder[i] = (dy / dx) * 0.00001

    if visualisation is True:
        plt.plot(time, current,'r')
        plt.plot(time[0:len(der)],der,'g')
        plt.plot(time[0:len(derder)], derder, 'm')
        plt.plot(time[0:len(derderder)],derderder,'lime')

        #plt.axvline(x=time[f_peak_index],color = 'k')
        #plt.axvline(x=time[der_peaks[1]],color = 'c')
        #plt.axvline(x=time[derder_peaks[1]],color = 'y')

        plt.grid()
        plt.show()




    charge = 0
    return charge


##################################################################################################################

#TEST THE FUNCTIONS

#CVs for each concentration
CV_10 = getdata_addconc("/Users/nathanrobins/Documents/MSc_Neurotech/MSc_Project/Data/In_vitro_basal_5-HT/101418/electrode1/10nm/100_CV.txt",10.0)
CV_25 = getdata_addconc("/Users/nathanrobins/Documents/MSc_Neurotech/MSc_Project/Data/In_vitro_basal_5-HT/101418/electrode1/25nm/100_CV.txt",25.0)
CV_50 = getdata_addconc("/Users/nathanrobins/Documents/MSc_Neurotech/MSc_Project/Data/In_vitro_basal_5-HT/101418/electrode1/50nm/100_CV.txt",50.0)
CV_100 = getdata_addconc("/Users/nathanrobins/Documents/MSc_Neurotech/MSc_Project/Data/In_vitro_basal_5-HT/101418/electrode1/100nm/100_CV.txt",100.0)

plot = False
while plot is True:
    plt.subplot(221)
    plt.plot(CV_10[1:,2])

    plt.subplot(222)
    plt.plot(CV_25[1:,2])

    plt.subplot(223)
    plt.plot(CV_50[1:,2])

    plt.subplot(224)
    plt.plot(CV_100[1:,2])

    plt.show()
    plot = False


#int_2(CV_10,visualisation=True)
#int_2(CV_100,visualisation=True)
some_int(CV_10)
some_int(CV_100)