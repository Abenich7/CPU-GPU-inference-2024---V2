import numpy as np 
import matplotlib.pyplot as plt

#model=train_model(model,train_loader,learning_rate=1e-4,num_epochs=30)
#print("Trainig complete.")

data=np.load('./training_history.npy',allow_pickle=True)

training_history1=data.item()
#print(training_history1)
a=training_history1['epoch'][0]
print(a)

b=training_history1['epoch'][-1]
print(b)
x=np.arange(a,b+1,1)

print(x)

y=training_history1['train_loss']
y=y[0:20]
print(y)


 
plt.xlabel('epoch')
plt.ylabel('train_loss')

plt.title('res18 train loss over epochs')

plt.plot(x,y)

plt.show()