from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torchaudio as audio
import os

class SoundData(Dataset):
    def __init__(self):
        self.no_snap = [(audio.load("NoSnap/"+elem)[0],torch.FloatTensor([0,1])) for elem in os.listdir("NoSnap")]

        self.snap = [(audio.load("Snap/"+elem)[0],torch.FloatTensor([1,0])) for elem in os.listdir("Snap")]
        self.data = self.no_snap + self.snap

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
