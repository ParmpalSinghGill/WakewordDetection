from tensorflow.keras.models import load_model

model=load_model("model.h5")
checkpoint_filepath = 'Models/'
model.load_weights(checkpoint_filepath)
print(model.summary())


