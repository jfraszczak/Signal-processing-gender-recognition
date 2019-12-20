import numpy as np
import librosa
from Model import get_MFCC_features
import pickle
import matplotlib.pyplot as plt
from scipy import stats

def predict(file):
    y, sr = librosa.load(file)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=80, fmax=250)
    if len(pitches[np.nonzero(pitches)]) > 0:
        pitch = np.mean(pitches[np.nonzero(pitches)])
        skew = stats.skew(pitches[np.nonzero(pitches)])
        flag = True
    else:
        flag = False

    if flag:
        if skew > 0.1 or pitch < 165:
            return 'M'
        elif skew < -0.2 or pitch > 175:
            return 'K'
        else:
            features = get_MFCC_features(file)
            model = pickle.load(open('model_evaluate', 'rb'))
            predicted = model.predict([features])
            if predicted[0] == 0:
                return 'K'
            else:
                return 'M'
    else:
        features = get_MFCC_features(file)
        model = pickle.load(open('model', 'rb'))
        predicted = model.predict([features])
        if predicted[0] == 0:
            return 'K'
        else:
            return 'M'

#print(predict('train/089_M.wav'))