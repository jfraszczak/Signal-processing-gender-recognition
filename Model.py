import scipy.io.wavfile
import numpy as np
from python_speech_features import mfcc
from sklearn import preprocessing
import os
from sklearn import ensemble
import pickle
import librosa

def mean_vector(frames):
    sums = np.zeros(len(frames[0]))
    for frame in frames:
        for i in range(len(frame)):
            sums[i] += frame[i]
    avg = []
    for i in range(len(sums)):
        n = len(frames)
        avg.append(sums[i] / n)

    return avg

def get_MFCC_features(file):
    sr, audio = scipy.io.wavfile.read(file)
    np.asarray(audio)
    if len(audio.shape) == 2:
        audio = audio[:, 0]
    features = mfcc(audio, sr, numcep=20, nfft = 1300)
    features = mean_vector(features)

    y, sr = librosa.load(file)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=80, fmax=250)
    if len(pitches[np.nonzero(pitches)]) > 0:
        pitch = np.mean(pitches[np.nonzero(pitches)])
    else:
        pitch = 0
    features.append(pitch)
    features = preprocessing.scale(features)
    return features

def model():
    files = []
    Y = []

    for file in os.listdir("train"):
        if file.endswith(".wav"):
            files.append('train/' + file)
            if file[4] == 'K':
                Y.append(0)
            else:
                Y.append(1)

    features = []
    for file in files:
        vector = get_MFCC_features(file)
        features.append(vector)
    np.asarray(features)

    model = ensemble.RandomForestClassifier(1000, 'entropy')
    model.fit(features, Y)

    pickle.dump(model, open('model', 'wb'))