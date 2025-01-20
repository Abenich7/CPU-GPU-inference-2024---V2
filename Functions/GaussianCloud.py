# Requires numpy, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class GaussianCloud:
        
    def __init__(self,label=None):
        self.num_features = None
        self.mean = None
        self.cov = None
        self.num_samples = None
        self.X = None
        self.label=label

    def create(self):
        self.num_features = int(input("How many features (ie random variables)?"))
        self.mean = eval(input(f"Input the means for {self.num_features} features (e.g., [-3,-2] for 2 features): "))
        self.cov = eval(input(f"Input covariance matrix for {self.num_features} features (e.g., [[1,0],[0,1]]): "))
        self.num_samples = int(input("Input number of samples: "))
        # Generate random samples
        
        rng = np.random.default_rng()
        self.X = rng.multivariate_normal(self.mean, self.cov, self.num_samples)
        
        print(f"Shape of self.X: {self.X.shape}")
        print(f"First few values of self.X: {self.X[:5]}")


       
    def add_plot_data(self, ax=None):
        if self.X is not None:
            if self.X.shape[1] >= 2:  # Ensure there are at least two features to plot
                x_coords = self.X[:, 0]  # First column (x-coordinates)
                y_coords = self.X[:, 1]  # Second column (y-coordinates)
                ax.scatter(x_coords, y_coords, label=f"Cloud ({self.mean})", alpha=0.6)
            else:
                print("Not enough features to plot a 2D cloud.")
        else:
            print("No data available. Please generate data first using the create() method.")

        
    @staticmethod
    def plot_all(clouds):
        fig, ax = plt.subplots()

        for cloud in clouds:
            cloud.add_plot_data(ax)  # Add data from each cloud to the plot
        
        ax.set_title('Multiple Gaussian Clouds')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.axis('equal')
        ax.legend()
        plt.show()
    
    

if __name__ == "__main__":
    from GaussianCloud import GaussianCloud
    clouds = []  # List to store cloud objects

    num_clouds = int(input("How many clouds?"))
    # Create and add multiple clouds
    for i in range(num_clouds):  # Creating 3 clouds for example
        cloud = GaussianCloud(label=i)
        cloud.create()
        clouds.append(cloud)

    # Now plot them all at once
    GaussianCloud.plot_all(clouds)


    # Combine data from all clouds into one dataset
    X_train = np.hstack([cloud.X for cloud in clouds]).T
    y_train = np.concatenate([[cloud.label] * cloud.num_samples for cloud in clouds])

    # Train a logistic regression model for multi-class classification
    model = LogisticRegression(multi_class='ovr', max_iter=1000)
    model.fit(X_train, y_train)

    # Plot all clouds
   # plt.figure(figsize=(8, 6))
    #for cloud in clouds:
     #   cloud.plot()

    # Plot the decision boundaries
    x_vals = np.linspace(-6, 8, 300)
    y_vals = np.linspace(-6, 8, 300)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict for each grid point to visualize decision regions
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2, cmap="viridis")

    plt.legend()
    plt.title("Gaussian Clouds with Logistic Regression Boundaries")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    # Predict labels and evaluate accuracy
    y_pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")






            

            
