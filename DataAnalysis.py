import numpy as np
from datasets import load_dataset, ClassLabel, Value
from Utils import preprocessData
from matplotlib import pyplot as plt

def getmfccs(numofclasses=100,numberofInstances=10):
    ds = load_dataset("speech_commands","v0.02",split="test")
    num_classes,names=ds.features["label"].num_classes,ds.features["label"].names
    numofclasses=min(num_classes,numofclasses)
    ds = ds.map(preprocessData)
    features=[]
    for i in range(numofclasses):
        ds1 = ds.filter(lambda x: x["label"] == i)
        print(i,names[i],ds1.num_rows)
        features.append((ds1[:numberofInstances]["audiomfccs"]))
    return np.array(features),names[:numofclasses]

def drawMfccs(feature,name):
    print(feature.shape,name)
    f, axarr = plt.subplots(*feature.shape[:2])
    for i in range(feature.shape[0]):
        for j in range(feature.shape[1]):
            print(feature[i,j].shape)
            axarr[i,j].imshow(feature[i,j].T)
        axarr[i, 0].set(ylabel=name[i])

    plt.show()

features,names=getmfccs(36,10)
drawMfccs(features,names)
