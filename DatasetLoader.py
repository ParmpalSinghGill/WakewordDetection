import datasets,time,scipy
import soundfile as sf
from datasets import load_dataset
from Utils import convertSampleRate





# ds = load_dataset("speech_commands","v0.01", split="validation")
ds = load_dataset("speech_commands","v0.02", split="validation")
print(ds._info.features["label"])
ds.set_transform(convertSampleRate)
# ds[0]
# print(ds[0])
# print(ds[0]["label"])
print(dir(ds))
print(ds[0])


# sf.write("test.wav",ds[1]["audio"]["array"],ds[0]["audio"]["sampling_rate"])
sf.write("test.wav",ds[1]["audio"]["array"],8000)



