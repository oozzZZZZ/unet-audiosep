import os
from tqdm import tqdm
import random
import numpy as np
from glob import glob
import datetime

from librosa.util import find_files
from librosa.core import load,stft,resample
import librosa

import parameter as C

PATH_FFT = C.PATH_FFT
SPEECH_PATH = C.TARGET_PATH
NOISE_PATH = C.NOISE_PATH

PATCH_LENGTH=C.PATCH_LENGTH

aug = C.augmentation

times = C.datatimes

def stretch(data, rate=1):
    input_length = len(data)
    data = librosa.effects.time_stretch(data, rate)
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data

def time_shift(data,shift):
    data_roll = np.roll(data, shift)
    return data_roll

def pitch_shift(data,sample_rate,shift):
    ret = librosa.effects.pitch_shift(data, sample_rate, shift, bins_per_octave=12, res_type='kaiser_best')
    return ret

def loadAudio(filename,augmentation=True):
    data, sr = load(filename, sr=None)
    if sr != C.SR:
        data = resample(data, sr, C.SR)
        
    if augmentation:
        """
        AUGMENTATION
        1. Amp
        2. Stretch
        3. Time Shift
        4. Pitch Shift
        """
        # 1. Amplitude
        data = data * random.uniform(0.8, 1.2)

        # 2. Stretch
        data = stretch(data, rate=random.uniform(0.8, 1.2))

        # 3. Time Shift
        data = time_shift(data,int(random.uniform(2**10, 2**13)))

        # 4. Pitch Shift
        data = pitch_shift(data,C.SR,random.uniform(-12, 12))
    
    # waveform -> Spec
    data = stft(data, n_fft=C.FFT_SIZE, hop_length=C.H, win_length=C.FFT_SIZE)
    return data

#main
def main():

    if not os.path.exists(PATH_FFT):
        os.mkdir(PATH_FFT)
    target_list = find_files(SPEECH_PATH, ext="wav")
    noise_list = find_files(NOISE_PATH, ext="wav")

    noise_num = len(noise_list)

    for time in tqdm(range(times),leave=True):
        random.shuffle(target_list)
        random.shuffle(noise_list)

        for target in tqdm(target_list,leave=False):

            target=loadAudio(target,augmentation=aug)
            noise=loadAudio(noise_list[random.randint(0, noise_num-1)],augmentation=aug)

            length=min(target.shape[1],noise.shape[1])

            target,noise=target[:,:length],noise[:,:length]
            addnoise = target+noise

            target = np.abs(target)
            target /= np.max(target)
            addnoise = np.abs(addnoise)
            addnoise /= np.max(addnoise)

            iterate = 0
            now = datetime.datetime.now()
            while length > PATCH_LENGTH:
                save_target = target[:,PATCH_LENGTH * iterate:PATCH_LENGTH * (iterate+1)]
                save_data = addnoise[:,PATCH_LENGTH * iterate:PATCH_LENGTH * (iterate+1)]
                fname = now.strftime('%Y%m%d%H%M%S') + "_" + str(iterate)
                np.savez(os.path.join(PATH_FFT, fname+".npz"),target=save_target, data=save_data)
                length-=PATCH_LENGTH
                iterate+=1
             
if __name__ == "__main__":
    main()
