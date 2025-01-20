import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def create_data_fn

np.random.seed(42)

#create 3 different batches of data that are close to each other. 
#for exmaple, can be sets of points that are close to each other
#create sets of random variables from a gaussian distribution 
x1=np.random.multivariate_normal(3,1,300)
x2=np.random.multivariate_normal(5,1,300)
x3=np.random.multivariate_normal(7,1,300)
X=np.vstack(x1,x2,x3)
y=np.array([0]*300+[1]*300+[2]*300)

train_ratio=0.77

X_train,X_test,y_train,y_test=train_test_split(X,y,train_ratio)

model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)

model.fit(X_train,y_train)
fig, ax = plt.subplots(figsize=(8, 6))




#try out different models on the data