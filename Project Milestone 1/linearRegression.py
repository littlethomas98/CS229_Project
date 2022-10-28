import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def importData():
    data = pd.read_csv('CleanMergedData.csv')
    data = data.to_numpy()
    return data

data = importData()

#Prepare Data 
#We should add distance as another feature of X
X_numpy = data[:,3].reshape(data.shape[0],1)
y_numpy = data[:,2].reshape(data.shape[0],1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

#Define Model
input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)

#Calculate Loss using MSE and optimization via Stochastic Gradient Descent
#(These are super basic models, but using for now since they are quick to implement)
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

#Training
num_epochs = 100
for epoch in range(num_epochs):
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    if (epoch+1 % 10 ==0):
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

#Plot
predicted = model(X).detach().numpy()
fig = plt.figure()
ax1 = fig.add_subplot()
plt.plot(X_numpy, y_numpy, 'o')

plt.plot(X_numpy, predicted)
ax1.set_xlabel('Earthquake Magnitude')
ax1.set_ylabel('SO2 Concentraion')
ax1.set_title('Linear Regression Model btw EQ Magnitude and SO2 concentration')
plt.show()