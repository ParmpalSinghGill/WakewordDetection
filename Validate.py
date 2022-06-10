from Utils import loadData
from keras.models import load_model

batch_size=32
# train_dataset, inputshape, num_classes = loadData("train", batch_size=batch_size)
# valid_dataset, inputshape, nu/media/parmpal/Workspace/python/Speech/WakeupWorddetection/modelTraind_85.h5m_classes = loadData("validation", batch_size=batch_size)
# test_dataset, inputshape, num_classes = loadData("test", batch_size=batch_size)
model=load_model("modelTraind_85.h5")
model.summary()
# model.evaluate(train_dataset)
# model.evaluate(valid_dataset)
# model.evaluate(test_dataset)
