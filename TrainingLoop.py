import torch
import torch.nn as nn
import torch.optim as optim
from SmallerNet import SmallerNet
from DataLoader import SoundData
from torch.utils.data import DataLoader

net = SmallerNet()
net.load_state_dict(torch.load("modell.pth"))
dataset = SoundData()
dataloader = DataLoader(dataset, 8, shuffle = True, drop_last = True)

lr = 0.01
epochs = 10

loss_fn = nn.BCELoss()
optimizer = optim.Adam(net.parameters())

for epoch in range(epochs):
    loss_list = []
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = net(X)
        #print(pred)
        loss = loss_fn(pred, y)
        loss_list.append(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    mean_loss = sum(loss_list)/len(loss_list)
    print("Loss over epoch {0}: ".format(epoch), mean_loss)

torch.save(net.state_dict(), "./modell.pth")
