#########################################################################
#  Question 3 on the assignement : the goal is to run Knn for different 
#values of  k and plot the accuracy for each value to choose the best
#hyper_parameter  #
#######################################################################

import utils
from run_knn import run_knn 
import plot_digits
import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":
    

    #loading the dataset
    train_data, train_labels = utils.load_train()

    #loading the validation set
    valid_data,valid_labels = utils.load_valid()


    # vector of each k
    K = np.array([1,3,5,7,9])
    
    #dictionnay result
    results={}

    for k in K:
        
        #prediction 
        prediction = run_knn(k,train_data,train_labels,valid_data)

        #computing the precision
        results[k]= np.mean(prediction==valid_labels)
    
    #plotting the result
    precisions = np.array([(k,results[k]) for k in results])
    plt.plot(precisions[:,0], precisions[:,1],'r-o')
    plt.title('precision as a function of k')
    plt.savefig("precisons_k.png")




