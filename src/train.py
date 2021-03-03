# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 14:42:13 2021

@author: yoka
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

def doTraining(epoch, model, criterion, optimizer, dataloader, printStride, modleName, saveDir):
    runningLoss = 0.0
    for i, (inputs, labels) in enumerate(dataloader, 0):
        optimizer.zero_grad()
        # 1.Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # 2.Backward
        loss.backward()
        # 3.Update
        optimizer.step()
        # Save training result\
        torch.save(model.state_dict(), saveDir + modleName + '-epoch-' + '{:03d}'.format(epoch) + '.pt')
        # Print traing process
        runningLoss += loss.item()
        if i % printStride == printStride - 1:
            print('[%d, %5d]  loss: %.2f'%(epoch + 1, i + 1, runningLoss/printStride))
            runningLoss = 0.0
  
def doTesting(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracyRate = 100 * correct / total
    print('Accuracy rate on test-dataset: %.2f %% (%d / %d)'%(accuracyRate, correct, total))
    return accuracyRate

def imgClassfierTraining(dataset, model, model_name, epoch, savepath='../save/', printStride=100):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    rates = []
    epochs = []
    for i in range(epoch):
        print('--------------------------')
        print('[epcho, batchIdx] loss-value')
        doTraining(i, model, criterion, optimizer,
                   dataset.trainDataloader, printStride, model_name, savepath)
        rate = doTesting(model, dataset.testDataloader)
        epochs.append(i)
        rates.append(rate)
    plt.plot(epochs, rates)
    plt.xlabel('epoch')
    plt.ylabel('accuracy (%)')
    plt.savefig(savepath + model_name + '.png')
    plt.show()

#from ykModels import FullConnectedNetModel
#from ykDataloader import MnistDataLoader
from ykModels import ConvolutionalNetModel
from ykDataloader import Cifa10DataLoader
#from ykModels import InceptionConvolutionalNetModel
#from ykModels import ResidualConvolutionalNetModel
if __name__ == '__main__':
    cifa = Cifa10DataLoader(64, '../', download=False)
    img = np.array(cifa.trainImages[0][0])
    w = img.shape[0]
    h = img.shape[1]
    imgClassfierTraining(cifa,                                  # dataset
                         ConvolutionalNetModel(3, w, h, 10),    # model
                         'cifa10-convnet',                      # model name
                         10)                                    # epoch