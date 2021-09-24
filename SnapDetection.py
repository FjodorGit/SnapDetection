#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 20:54:00 2021

@author: don
"""
from scipy.io.wavfile import write
import sounddevice as sd
import numpy as np
import torch
from SmallerNet import SmallerNet

net = SmallerNet()

net.load_state_dict(torch.load("modell.pth"))
net.eval()

stream = sd.InputStream(samplerate=16000, channels=1)
stream.start()
data = np.zeros((1,5000))
for i in range(100):
    next_frame = stream.read(2500)[0]
    next_frame = next_frame.reshape((-1,2500))
    #print(data.shape)
    #print(next_frame.shape)
    data = np.concatenate((data[:,-2500:],next_frame), axis = 1)
    #print(torch.from_numpy(data).unsqueeze(0).shape)
    pred = net(torch.from_numpy(data).view(-1,1,5000).type(torch.FloatTensor))
    if pred[0][0].item() >= 0.9:
        print("Snap!!!")
    #write('NoSnap/output_{0}.wav'.format(i), 16000, next_frame)


# =============================================================================
# fs = 44100  # Sample rate
# seconds = 3  # Duration of recording
#
# myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
# sd.wait()  # Wait until recording is finished
# write('output.wav', fs, myrecording)  # Save as WAV file
# =============================================================================
