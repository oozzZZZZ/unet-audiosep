import pandas as pd
import os
from librosa.core import load,stft,resample,istft
import numpy as np
import torch

import parameter as C
import network

from IPython.display import display, Audio



def denoiser(audio_data,model_path,hard_rate=0.9):

    model = network.UnetConv2()
    model.load_state_dict(torch.load(model_path))

    spec = stft(audio_data, n_fft=C.FFT_SIZE, hop_length=C.H, win_length=C.FFT_SIZE)

    mag = np.abs(spec)
    mag /= np.max(mag)
    phase = np.exp(1.j*np.angle(spec))

    full_len=mag.shape[1]
    times, mod = divmod(full_len, C.PATCH_LENGTH * C.BATCH_SIZE)
    
    if not times==0:
        for i in range(times):
            process = mag[:,C.PATCH_LENGTH * C.BATCH_SIZE * i:C.PATCH_LENGTH * C.BATCH_SIZE * (i+1)]
            listen_list=[]
            for iterate in range(C.BATCH_SIZE):
                _data = process[:,C.PATCH_LENGTH * iterate:C.PATCH_LENGTH * (iterate+1)]
                _data=torch.from_numpy(_data.astype(np.float32)).clone()
                listen_list.append(_data)
            tensor_data = torch.stack(listen_list)
            mask=model(tensor_data)
            mask[mask < hard_rate]=0
            h=tensor_data * mask
            h = h.to('cpu').detach().numpy().copy()

            if i==0:
                output = h[0,:,:]
                for f in range(1,C.BATCH_SIZE):
                    output = np.concatenate([output, h[f,:,:]], 1)

            else:
                for f in range(C.BATCH_SIZE):
                    output = np.concatenate([output, h[f,:,:]], 1)

        if not mod==0:            
            process = mag[:,-mod:]
            addempty=np.zeros([mag.shape[0],C.PATCH_LENGTH * C.BATCH_SIZE - mod])
            process = np.concatenate([process,addempty], 1)

            listen_list=[]
            for iterate in range(C.BATCH_SIZE):
                _data = process[:,C.PATCH_LENGTH * iterate:C.PATCH_LENGTH * (iterate+1)]
                _data=torch.from_numpy(_data.astype(np.float32)).clone()
                listen_list.append(_data)
            tensor_data = torch.stack(listen_list)
            mask=model(tensor_data)
            mask[mask < hard_rate]=0
            h=tensor_data * mask
            h = h.to('cpu').detach().numpy().copy()
            for f in range(C.BATCH_SIZE):
                output = np.concatenate([output, h[f,:,:]], 1)
                
    else:
        process = mag[:,-mod:]
        addempty=np.zeros([mag.shape[0],C.PATCH_LENGTH * C.BATCH_SIZE - mod])
        process = np.concatenate([process,addempty], 1)

        listen_list=[]
        for iterate in range(C.BATCH_SIZE):
            _data = process[:,C.PATCH_LENGTH * iterate:C.PATCH_LENGTH * (iterate+1)]
            _data=torch.from_numpy(_data.astype(np.float32)).clone()
            listen_list.append(_data)
        tensor_data = torch.stack(listen_list)
        mask=model(tensor_data)
        mask[mask < hard_rate]=0
        h=tensor_data * mask
        h = h.to('cpu').detach().numpy().copy()
        
        output = h[0,:,:]
        for f in range(1,C.BATCH_SIZE):
            output = np.concatenate([output, h[f,:,:]], 1)       

    denoise=istft(output[:,:phase.shape[1]]*phase,hop_length=C.H, win_length=C.FFT_SIZE)
    
    return denoise



###################main###################
def main():
    df_noise=pd.read_csv("/disk107/raw_data/unet_speech/type_noise.csv",sep="\t",header=None)
    df_voice=pd.read_csv("/disk107/raw_data/unet_speech/type_utt.csv")
    file_path = "/disk107/DATA/ERNIE/WAV"

    model_path = "./model/model_20210208_060155_Epoch260.pt"
    hard_rate = 0.90

    file_list=df_voice["0"].unique()

    filename = file_list[2]
    filepath=os.path.join(file_path, filename+".wav")
    data, sr = load(filepath, sr=None)
    if not sr==C.SR:
        data = resample(data, sr, C.SR)

    print("処理前音声")
    display(Audio(data, rate=sr))

    denoise=denoiser(data,model_path,hard_rate)
    print("処理後音声")
    display(Audio(denoise, rate=16000))

if __name__ == "__main__":
    main()
