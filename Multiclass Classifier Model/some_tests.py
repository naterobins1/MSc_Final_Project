from ML.ML_functions import load_all_data
from Visualisation.visualisation import CV_plot, number_concentrations
import matplotlib.pyplot as plt

CV_d,CV_l = load_all_data(include_time=True,catergor=False,normalise=True)
print(len(CV_d))
#CV_plot(CV_d,CV_l,[10,25,50,75,100],1)

no_conc = number_concentrations(CV_l,show_number=True,show_plot=False, catergorical=False)


#jobub=no_conc/sum(no_conc)

#print(jobub)
#labels = ['10', '25', '50', '75', '100']
#plt.xticks(range(len(jobub)), labels)
#plt.xlabel('Serotonin Concentration (nm)')
#plt.ylabel('Normalised Frequency')
##plt.title('I am title')
#plt.bar(range(len(jobub)), jobub)
#plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y')
#plt.show()

