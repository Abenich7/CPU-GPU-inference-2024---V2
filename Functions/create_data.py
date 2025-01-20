import numpy as np
import matplotlib.pyplot as plt 

def create_data_fn(num_samples,num_classes,means,covs):

  
    samples_per_class = num_samples // num_classes
    X = []
    y = []
    
    for i in range(num_classes):
        X_class = np.random.multivariate_normal(mean=means[i], cov=covs[i], size=samples_per_class)
        y_class = np.full(samples_per_class, i)
        print(X_class.shape)
        X.append(X_class)
        y.append(y_class)


    X = np.vstack(X)
    y = np.hstack(y)
    
    # Plot each class in a different color
   # plt.figure(figsize=(8, 6))
   # for i in range(num_classes):
    #    X_class = X[y == i]  # Extract points belonging to class `i`
     #   plt.scatter(X_class[:, 0], X_class[:, 1], label=f"Class {i}")  # Scatter plot

  
   # plt.title("DATA")
   # plt.grid(True)
   # plt.show()



    return X,y


