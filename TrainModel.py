import datasets,time,scipy
import soundfile as sf
import tensorflow as tf
import numpy as np
from datasets import load_dataset, ClassLabel, Value
from Utils import preprocessData,getModel

batch_size=32
def Collator(ds):
    labeles,feautres=[],[]
    for d in ds:
        labeles.append(d["label"])
        feautres.append(np.reshape(np.concatenate(d["audiomfccs"]),(-1,len(d["audiomfccs"][0]),1)))
    feautres,labeles= np.array(feautres),np.array(labeles)
    feautres,labeles= tf.convert_to_tensor(feautres),tf.convert_to_tensor(labeles)
    # return feautres,labeles
    return {"features":feautres,"label":labeles}

def loadData(split="test"):
    # ds = load_dataset("speech_commands","v0.01", split="validation")
    ds = load_dataset("speech_commands","v0.02",split=split)
    # print(ClassLabel(ds))
    print(ds.features)
    # ds.set_transform(preprocessData)
    ds=ds.filter(lambda x:x["audio"]["array"].shape[0]==16000)
    ds=ds.map(preprocessData)
    train_dataset = ds.to_tf_dataset(
        columns=['audiomfccs'],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=Collator,
        label_cols="label"
    )
    inputshape=(*np.array(ds[0]["audiomfccs"]).shape,1)
    num_classes=ds.features["label"].num_classes
    return train_dataset,inputshape,num_classes

def TrainModel(model,train_dataset,valid_dataset):
    model.fit(train_dataset,epochs=20,validation_data=valid_dataset)
    model.save("model.h5")
    # print(type(train_dataset))
    # d,l=next(iter(train_dataset))
    # print(d.shape)
    # print(l.shape)


if __name__ == '__main__':
    # loadData()
    train_dataset,inputshape,num_classes=loadData("train")
    valid_dataset,_,_=loadData("validation")
    model=getModel(inputshape,num_classes)
    TrainModel(model,train_dataset,valid_dataset)




# numlabel=
# print(ds[0:10]["audiomfccs"].shape)
# print(ds[""])



#
# print(dir(ds))
# print(type(ds))
# # print(ds[0])
# # print()
# print(ds.to_pandas().columns)
# print(ds._indices)

# new_features = ds.features.copy()
# new_features['label'] = ClassLabel(num_classes=36)
# # new_features['mffcs'] = Value('mffcs')
# new_features['audio'] = Value('audio')
# ds = ds.cast(new_features,batch_size=2)
# print(ds.features)

# # sf.write("test.wav",ds[1]["audio"]["array"],ds[0]["audio"]["sampling_rate"])
# sf.write("test.wav",ds[1]["audio"]["array"],8000)



