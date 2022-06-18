import time

import sounddevice as sd
from scipy.io.wavfile import write
import pyaudio,wave
import numpy as np

def SaveAudio():
    fs = 44100  # Sample rate
    seconds = 3  # Duration of recording

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('output.wav', fs, myrecording)  # Save as WAV file

def saveAudioStremForSecond(seconds=4):

    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2
    fs = 44100  # Record at 44100 samples per second
    filename = "output.wav"

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()


def saveAudioStrem():
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2
    fs = 44100  # Record at 44100 samples per second
    filename = "output.wav"
    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    try:
        while True:
            data = stream.read(chunk)
            frames.append(data)
    except KeyboardInterrupt as ke:
        pass
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

def saveAudioStremNpOneChannel(seconds=4):
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44100 samples per second
    filename = "output1.wav"
    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    try:
        # while True:
        for i in range(0, int(fs / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(np.frombuffer(data, np.int16))
    except KeyboardInterrupt as ke:
        pass
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')
    print(len(frames),frames[0].shape)
    frames=np.concatenate(frames,axis=0)
    print(frames.shape)
    write(filename, fs, frames.astype(np.int16))

def saveAudioStremNp(seconds=4,channels = 2):
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    fs = 44100  # Record at 44100 samples per second
    filename = "output1.wav"
    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    print(fs / chunk * seconds,fs , chunk , seconds)
    try:
        # while True:
        for i in range(0, int(fs / chunk * seconds)):
            data = stream.read(chunk)
            if channels==1:
                frames.append(np.frombuffer(data, np.int16))
            else:
                frames.append(np.frombuffer(data, np.int16).reshape(-1,2))
    except KeyboardInterrupt as ke:
        pass
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')
    print(len(frames),frames[0].shape)
    frames=np.concatenate(frames,axis=0)
    print(frames.shape)
    write(filename, fs, frames.astype(np.int16))

def saveAudioStremNpExperiment(seconds=4,channels = 1):
    fs = 44100  # Record at 44100 samples per second
    chunk = fs  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    filename = "output1.wav"
    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    try:
        i=0
        print(fs / chunk * seconds)
        while True:
            if seconds and i >(fs / chunk * seconds): break
            i+=1
            data = stream.read(chunk)
            if channels==1:
                frames.append(np.frombuffer(data, np.int16))
            else:
                frames.append(np.frombuffer(data, np.int16).reshape(-1,2))

    except KeyboardInterrupt as ke:
        pass
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')
    print(len(frames),frames[0].shape)
    frames=np.concatenate(frames,axis=0)
    print(frames.shape)
    write(filename, fs, frames.astype(np.int16))



# SaveAudio()
# saveAudioStremForSecond()
saveAudioStremNpExperiment()
