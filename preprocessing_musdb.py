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

target = "vocals"

PATH_FFT = C.PATH_FFT
SPEECH_PATH = C.SPEECH_PATH
NOISE_PATH = C.NOISE_PATH

PATCH_LENGTH=C.PATCH_LENGTH

times = C.datatimes
musdb_path=C.MUSDB_PATH

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

    bass_list=glob(musdb_path+"/*/*/bass.wav", recursive=True)
    drums_list=glob(musdb_path+"/*/*/drums.wav", recursive=True)
    other_list=glob(musdb_path+"/*/*/other.wav", recursive=True)
    vocals_list=glob(musdb_path+"/*/*/vocals.wav", recursive=True)

    if target == "vocals":
        noise_list1,noise_list2,noise_list3 = bass_list,drums_list,other_list
        target_list = vocals_list

    if target == "bass":
        noise_list1,noise_list2,noise_list3 = drums_list , other_list , vocals_list
        target_list = bass_list

    if target == "drums":
        noise_list1,noise_list2,noise_list3 = bass_list , other_list , vocals_list
        target_list = drums_list

    if target == "other":
        noise_list1,noise_list2,noise_list3 = bass_list , drums_list , vocals_list
        target_list = other_list

    noise_num = len(noise_list1)

    random.shuffle(target_list)
    random.shuffle(noise_list1)
    random.shuffle(noise_list2)
    random.shuffle(noise_list3)

    for time in tqdm(range(times),leave=True):

        for target in tqdm(target_list,leave=False):

            target=loadAudio(target,augmentation=True)
            noise1=loadAudio(noise_list1[random.randint(0, noise_num-1)],augmentation=True)
            noise2=loadAudio(noise_list2[random.randint(0, noise_num-1)],augmentation=True)
            noise3=loadAudio(noise_list3[random.randint(0, noise_num-1)],augmentation=True)

            length=min(target.shape[1],noise1.shape[1],noise2.shape[1],noise3.shape[1])

            target,noise1,noise2,noise3=target[:,:length],noise1[:,:length],noise2[:,:length],noise3[:,:length]
            addnoise = target+noise1+noise2+noise3

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