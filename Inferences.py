import time,datetime
from tensorflow.keras.models import load_model
# import logging
import numpy as np
import sounddevice,pyaudio
from scipy.io.wavfile import write
from Utils import getMFCC
# logging.basicConfig(level=20)

# import python_speech_features
# num_mfcc = 16
# def getMFCC(data,samplerate):
#     mfccs = python_speech_features.base.mfcc(data,samplerate=samplerate,
#          winlen=0.256,winstep=0.050,numcep=num_mfcc,nfilt=26,
#          nfft=2048,preemph=0.0,ceplifter=0,appendEnergy=False,winfunc=np.hanning)
#     return mfccs


samplerate=8000 # Record at 44100 samples per second
chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
filename="Rec.wav"


def SaveAudio():
    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    print('Start recording')
    stream = p.open(format=sample_format,
                   channels=1,rate=samplerate,
                   frames_per_buffer=chunk,input=True)
    st,stt=time.time(),time.time()
    frames = []
    try:
        while True:
            data = stream.read(chunk)
            frames.append(np.frombuffer(data, np.int16))
    except KeyboardInterrupt as ke:
        pass
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    print('Finished recording')
    print(len(frames),frames[0].shape)
    frames=np.concatenate(frames,axis=0)
    print(frames.shape)
    write(filename, samplerate, frames.astype(np.int16))

model=load_model("h5models/modelTraind_85.h5")
labels=['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow', 'backward', 'forward', 'follow', 'learn', 'visual', '_silence_']
def predict(data):
    mfcc=getMFCC(np.concatenate(data,axis=0),samplerate)
    mfcc=np.reshape(mfcc,(1,*mfcc.shape,1))
    pred=np.argmax(model.predict(mfcc)[0])
    print(pred,labels[pred])

def Inference(eachsecond=.5):
    chunk=int(samplerate*eachsecond)
    print("chunk",chunk)
    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    print('Start recording')
    stream = p.open(format=sample_format,
                   channels=1,rate=samplerate,
                   frames_per_buffer=chunk,input=True)
    st,stt=time.time(),time.time()
    frames,framepersonds = [],samplerate//chunk
    print(framepersonds)
    try:
        while True:
            data = stream.read(chunk)
            frames.append(np.frombuffer(data, np.int16))
            if len(frames)>framepersonds:
                predict(frames[-framepersonds:])
            # print(datetime.datetime.now())
    except KeyboardInterrupt as ke:
        pass
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    print('Finished recording')
    print(len(frames),frames[0].shape)
    frames=np.concatenate(frames,axis=0)
    print(frames.shape)
    write(filename, samplerate, frames.astype(np.int16))



if __name__ == '__main__':
    # SaveAudio()
    Inference()

