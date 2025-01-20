
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
import time
import create_data
import test_model
import plot_ellipse
import numpy as np
import numpy as np
from matplotlib.patches import Ellipse
import GaussianCloud

means=input("Input the means for the gaussian cloud (2-variate) distribution, ex: [3,-3] :")
GaussianCloud.




num_classes=3
train_ratio = 0.8

# יצירת עננים גאוסיים מופרדים
np.random.seed(42)
means = [[-3, -3], [3, -3], [3, 3]]  # תוחלת לכל ענן
cov = [[0.8, 0], [0, 0.8]]  # מטריצת שונות


X,y=create_data.create_data_fn(num_samples,num_classes,means,cov)

# יצירת המודל
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)

# הרצת המודל
test_model.test_model_fn(X,y,train_ratio,model)

# ציור עננים
#plt.figure(figsize=(8, 6))
#for label, color in zip([0, 1, 2], ['red', 'green', 'blue']):
 #   plt.scatter(X[y == label][:, 0], X[y == label][:, 1], label=f'Class {int(label)}', alpha=0.6, edgecolor='k')
#plt.title("Three Non-Overlapping Gaussian Clouds")
#plt.xlabel("Feature 1")
#plt.ylabel("Feature 2")
#plt.legend()
#plt.grid(True)
#plt.show()



# ציור עננים
fig, ax = plt.subplots(figsize=(8, 6))
for label, color, mean_point in zip([0, 1, 2], ['red', 'green', 'blue'], means):
    ax.scatter(X[y == label][:, 0], X[y == label][:, 1], label=f'Class {int(label)}', alpha=0.6, edgecolor='k', color=color)
    # ציור אליפסה סביב הענן
    plot_ellipse.plot_cov_ellipse(means, cov, ax, n_std=2.0, edgecolor=color, facecolor='none', linewidth=2)

plt.title("Three Non-Overlapping Gaussian Clouds with Ellipses")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()


