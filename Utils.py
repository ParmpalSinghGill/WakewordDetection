import scipy.signal as sps
import python_speech_features
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from matplotlib import pyplot as plt
from datasets import load_dataset
from tensorflow.keras import models,layers,losses,optimizers


feature_sets_file = 'all_targets_mfcc_sets.npz'
perc_keep_samples = 1.0 # 1.0 is keep all samples
val_ratio = 0.1
test_ratio = 0.1
sample_rate = 8000
num_mfcc = 16
len_mfcc = 16


def getTfMFCC(signal,sample_rate):
    # A 1024-point STFT with frames of 64 ms and 75% overlap.
    stfts = tf.signal.stft(signal, frame_length=1024, frame_step=256,
                           fft_length=1024)
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1].value
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
      upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
      spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
      linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
      log_mel_spectrograms)[..., :13]
    return mfccs


def decimate(signal, old_fs, new_fs):
    number_of_samples = round(len(signal) * float(new_fs) / old_fs)
    return  sps.resample(signal, number_of_samples)
    # # Check to make sure we're downsampling
    # if new_fs > old_fs:
    #     print("Error: target sample rate higher than original")
    #     return signal, old_fs
    #
    # # We can only downsample by an integer factor
    # dec_factor = old_fs / new_fs
    # if not dec_factor.is_integer():
    #     print("Error: can only decimate by integer factor")
    #     return signal, old_fs
    # # Do decimation
    # resampled_signal = sps.decimate(signal, int(dec_factor))
    # return resampled_signal, new_fs

def plotmfccs(mffcs):
    for mfcc in mffcs:
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(mfcc, x_axis='time')
    plt.show()

def getMFCC(data,samplerate):
    mfccs = python_speech_features.base.mfcc(data,samplerate=samplerate,
         winlen=0.256,winstep=0.050,numcep=num_mfcc,nfilt=26,
         nfft=2048,preemph=0.0,ceplifter=0,appendEnergy=False,winfunc=np.hanning)
    # mfccs = python_speech_features.base.mfcc(data, samplerate=samplerate, numcep=num_mfcc)
    # MFCC1 = librosa.feature.mfcc(data, sr=samplerate, n_mfcc=num_mfcc)
    # plt.figure(figsize=(12, 4))
    # librosa.display.waveshow(data, sr=samplerate)
    # plotmfccs([mfccs,MFCC1])
    # print(data.shape, mfccs.shape, MFCC1.shape)
    return mfccs




# def convertSampleRate(ds,new_rate=8000):
#     for audio in ds["audio"]:
#         data = audio["array"]
#         sampling_rate = audio["sampling_rate"]
#         assert new_rate < sampling_rate, "upsampling is notpossible "
#         data = decimate(data, sampling_rate, new_rate)
#         audio["array"] = data
#         audio["sampling_rate"] = new_rate
#     return ds
#
# def preprocessData(ds):
#     convertSampleRate(ds, new_rate=sample_rate)
#     audios = ds["audio"]
#     mfccs=[]
#     for audio in audios:
#         mfccs.append(getMFCC(audio["array"], sample_rate))
#     ds["audiomfccs"] = np.expand_dims(mfccs,axis=-1)
#     return ds

def convertSampleRate(ds,new_rate=8000):
    data = ds["audio"]["array"]
    sampling_rate =  ds["audio"]["sampling_rate"]
    assert new_rate < sampling_rate, "upsampling is notpossible "
    data = decimate(data, sampling_rate, new_rate)
    ds["audio"]["array"] = data
    ds["audio"]["sampling_rate"] = new_rate
    return ds

def preprocessData(ds):
    convertSampleRate(ds, new_rate=sample_rate)
    # ds["audiomfccs"] = np.expand_dims(getMFCC(ds["audio"]["array"], sample_rate),axis=-1)
    ds["audiomfccs"] = getMFCC(ds["audio"]["array"], sample_rate)
    return ds


def Collator(ds):
    labeles,feautres=[],[]
    for d in ds:
        labeles.append(d["label"])
        feautres.append(np.reshape(np.concatenate(d["audiomfccs"]),(-1,len(d["audiomfccs"][0]),1)))
    feautres,labeles= np.array(feautres),np.array(labeles)
    feautres,labeles= tf.convert_to_tensor(feautres),tf.convert_to_tensor(labeles)
    # return feautres,labeles
    return {"features":feautres,"label":labeles}

def loadData(split="test",batch_size=32):
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


def getModel(input_shape,nclasses):
    # Build model
    print(input_shape)
    model = models.Sequential()
    model.add(layers.Conv2D(32,
                            (2, 2),
                            activation='relu',
                            input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # model.add(layers.Conv2D(32, (2, 2), activation='relu'))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Classifier
    model.add(layers.Flatten())
    # model.add(layers.Dense(256, activation='relu'))
    # # model.add(layers.Dropout(0.2))
    # model.add(layers.Dense(512, activation='relu'))
    # # model.add(layers.Dropout(0.2))
    # model.add(layers.Dense(256, activation='relu'))
    # # model.add(layers.Dropout(0.2))
    # model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(nclasses, activation='softmax'))

    # Display model
    model.summary()

    # Add training parameters to model
    model.compile(loss=losses.sparse_categorical_crossentropy,
                  optimizer=optimizers.Adam(),
                  metrics=['acc'])

    return model