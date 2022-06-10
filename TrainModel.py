import datasets,time,scipy
import soundfile as sf
import tensorflow as tf
import numpy as np
from Utils import loadData,getModel

batch_size=32
def TrainModel(model,train_dataset,valid_dataset):
    checkpoint_filepath = 'Models/'
    model.save("model.h5")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_acc',
        mode='max',
        save_best_only=True)
    model.fit(train_dataset,epochs=40,validation_data=valid_dataset,callbacks=[model_checkpoint_callback])
    model.save("modelTraind.h5")
    # print(type(train_dataset))
    # d,l=next(iter(train_dataset))
    # print(d.shape)
    # print(l.shape)


if __name__ == '__main__':
    # loadData()
    train_dataset,inputshape,num_classes=loadData("train",batch_size=batch_size)
    valid_dataset,_,_=loadData("validation",batch_size=batch_size)
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



