# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 22:39:54 2020

@author: wnchang
"""

import os
#os.chdir('C:/Users/wnchang/Documents/F/PhD_Course/ECE629_NN/4.Project/Proj1')
import sys
import numpy as np
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.utils.data as Data

PI = 3.14
   
def main(): 
    
    torch.manual_seed(16)
    
#    # this is a Network
    class Net(torch.nn.Module):
        def __init__(self, n_feature, n_hidden, n_output):
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
            self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer
    
        def forward(self, x):
            x = F.relu(self.hidden(x))  # ReLu function for hidden layer
            x = self.predict(x)  # linear output
            return x
    
    
    n_feature = 1  # the number of input neurons
    n_hidden = 5000  # the number of hidden neurons
    n_output = 1  # the number of output neurons
    net = Net(n_feature, n_hidden, n_output)  # define the network
    print(net)  # net architecture
      
    
    x= torch.unsqueeze(torch.linspace(-24, 24, 1000),dim=1)  # x data (tensor), shape=(100, 1)
    y = torch.sin(PI * x/12)  + 100   
    #T=1
    #y = 100 + torch.cos(2*PI*(x/T)) # function in task2
    
    x, y = Variable(x), Variable(y)
#    plt.scatter(x.data.numpy(), y.data.numpy())
    
#    #another way to define a network
#    net = torch.nn.Sequential(
#            torch.nn.Linear(1, 5000),
#            torch.nn.ReLU(),
##            torch.nn.Linear(200, 100),
##            torch.nn.ReLU(),
#            torch.nn.Linear(5000, 1),
#        )
    
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
    
    BATCH_SIZE = 1000
    EPOCH = 3000
    PLOT_EPOCH = 500
    
    torch_dataset = Data.TensorDataset(x, y)
    
    loader = Data.DataLoader(
        dataset=torch_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=2,)
    
    #my_images = []
    #fig, ax = plt.subplots(figsize=(16,10))
    
    plt.figure(figsize=(10,10))
#    plt.figure(figsize=(10,10), dpi=80)
#    plt.figure(1)
#    figs = list()
    i=1    
    lossValue = []
    # start training
    start = time.time()
    for epoch in range(1, EPOCH+1):
        
        if epoch % 50 == 0:
                print("epoch: ", epoch )
        
        for step, (batch_x, batch_y) in enumerate(loader): # for each training step
            
            b_x = Variable(batch_x)
            b_y = Variable(batch_y)
    
            prediction = net(b_x)     # input x and predict based on x
    
            loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
            lossValue.append(loss)
    
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
            
            if epoch % PLOT_EPOCH == 0:
                # plot and show learning process
#                ax = "ax"+str(i)
#                ax = plt.subplot(3, 2, i)
                plt.subplot(3, 2, i)
                i+=1
                
#                z = np.sqrt(x**2)
#                plt.ylim(95, 105)
                plt.scatter(x.data.numpy(), y.data.numpy(), c='r', alpha=0.5, s=0.5)
#                plt.scatter(x.data.numpy(), y.data.numpy())
                plt.scatter(b_x.detach().numpy(), prediction.detach().numpy(), c='b',alpha=0.5, s=0.5)
                plt.legend(["y","y_hat"])
#                plt.plot(b_x.detach().numpy(), b_y.detach().numpy(), 'c')
#                plt.text(12, 95.3, 'Loss=%.4f' % loss.data.numpy(),fontdict={'size': 10,'color': 'red'})
                title = "Epoch:"+str(epoch)+" Hidden Neurons:"+str(1000)
                plt.title(title, fontsize="small", fontweight="bold")   
            
    end = time.time()
    print("Training time: ", end - start) 
            

    timestr = time.strftime("%Y%m%d-%H%M%S")
    imgName = "./Task1_" + timestr + ".png"
    plt.savefig(imgName)
    
    plt.figure(figsize=(7,7))
    imgLoss = "./Task_1_loss" + timestr + ".png"
    plt.plot(lossValue)
    plt.title("MSE Loss Value")
    plt.savefig(imgLoss)
    
    return 

if __name__ == "__main__":
    main()
    
    