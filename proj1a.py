# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 22:39:54 2020

@author: wnchang
"""

import os
os.chdir('C:/Users/wnchang/Documents/F/PhD_Course/ECE629_NN/4.Project/Proj1')
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
    n_hidden = 500  # the number of hidden neurons
    n_output = 1  # the number of output neurons
    net = Net(n_feature, n_hidden, n_output)  # define the network
    print(net)  # net architecture
      
    
    x= torch.unsqueeze(torch.linspace(-24, 24, 1000),dim=1)  # x data (tensor), shape=(100, 1)
    y = torch.sin(PI * x/12)  + 100   

    
    x, y = Variable(x), Variable(y)

    

    
    optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
    
#    BATCH_SIZE = 100
    EPOCH = 6000
    PLOT_EPOCH = 1500
    

    lossValue = []
    # start training
    start = time.time()
    for epoch in range(1, EPOCH+1):
        
        if epoch % 100 == 0:
                print("epoch: ", epoch )
        
#        for step, (batch_x, batch_y) in enumerate(loader): # for each training step
            
#        b_x = Variable(batch_x)
#        b_y = Variable(batch_y)
        b_x = x
        b_y = y

        prediction = net(b_x)     # input x and predict based on x

        loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
        lossValue.append(loss)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        
        if epoch % PLOT_EPOCH == 0:

            
            plt.figure(figsize=(6,6))
            plt.scatter(x.data.numpy(), y.data.numpy(), c='r', alpha=0.5, s=0.5)
            plt.scatter(b_x.detach().numpy(), prediction.detach().numpy(), c='b',alpha=0.5, s=0.5)
            title = "Epoch:"+str(epoch)+" Hidden Neurons:"+str(50)
            plt.title(title, fontsize="small", fontweight="bold")  
            plt.legend(["y","y_hat"])
            timestr = time.strftime("%Y%m%d-%H%M%S")
            imgName = "./Task1_" + timestr + ".png"
            plt.savefig(imgName)
            
    end = time.time()
    print("Training time: ", end - start) 
            

#    timestr = time.strftime("%Y%m%d-%H%M%S")
#    imgName = "./Task1_" + timestr + ".png"
#    plt.savefig(imgName)
    
    plt.figure(figsize=(7,7))
    imgLoss = "./Task_1_loss" + timestr + ".png"
    plt.plot(lossValue)
    plt.title("MSE Loss Value")
    plt.savefig(imgLoss)
    
    return 

if __name__ == "__main__":
    main()
    
    