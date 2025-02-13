import os
import numpy as np
import time
import sys
import torchvision.models as models
from ChexnetTrainer import ChexnetTrainer

#-------------------------------------------------------------------------------- 

def main ():
    
    #runTest()
    runTrain()
    #runTest()
  
#--------------------------------------------------------------------------------   

def runTrain():
    
    DENSENET121 = 'DENSE-NET-121'
    DENSENET169 = 'DENSE-NET-169'
    DENSENET201 = 'DENSE-NET-201'
    RESNET18 = 'RESNET-18'
    resnet18 = models.resnet18()
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    
    #---- Path to the directory with images
    pathDirData = '/pylon5/ac5616p/edgarxi/chestx/data'
    
    #---- Paths to the files with training, validation and testing sets.
    #---- Each file should contains pairs [path to image, output vector]
    #---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pathFileTrain = '/pylon5/ac5616p/edgarxi/chestx/chexnet/dataset/train_1.txt'
    pathFileVal = '/pylon5/ac5616p/edgarxi/chestx/chexnet/dataset/val_1.txt'
    pathFileTest = '/pylon5/ac5616p/edgarxi/chestx/chexnet/dataset/test_1.txt'
    
    #---- Neural network parameters: type of the network, is it pre-trained 
    #---- on imagenet, number of classes
    nnArchitecture = RESNET18
    nnIsTrained = True
    nnClassCount = 14
    
    #---- Training settings: batch size, maximum number of epochs
    trBatchSize = 32 #16
    trMaxEpoch = 100
    
    #---- Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = 256
    imgtransCrop = 224
        
    pathModel = 'm-' + timestampLaunch + '.pth.tar'
    
    print ('Training NN architecture = ', nnArchitecture)
    ChexnetTrainer.train(pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, None)
    
    print ('Testing the trained model')
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#-------------------------------------------------------------------------------- 

def runTest():
    
    pathDirData = '/pylon5/ac5616p/edgarxi/chestx/data'
    pathFileTest = '/pylon5/ac5616p/edgarxi/chestx/chexnet/dataset/test_1.txt'
    nnArchitecture = RESNET18
    nnIsTrained = True
    nnClassCount = 14
    trBatchSize = 16
    imgtransResize = 256
    imgtransCrop = 224
    
    pathModel = '/pylon5/ac5616p/edgarxi/chestx/chexnet/models/m-25012018-123527.pth.tar'
    
    timestampLaunch = ''
    print("loaded data, beginning test!")
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#-------------------------------------------------------------------------------- 

if __name__ == '__main__':
    main()





