from helpers import *
import os
import numpy as np
import cv2 as cv
from paths import *
from constants import *
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset 
from torchvision import transforms
from torchvision.ops import sigmoid_focal_loss
from sklearn.model_selection import train_test_split
from torchmetrics import F1Score

max_f1 = 0.95
faces_unknown = np.array([img for img in np.load(FACES_PATH + '/' + "unknown.npy")])
faces_unknown_zoomed_out = np.array([img for img in np.load(FACES_PATH + '/' + "unknown_zoomed_out.npy")])

print(faces_unknown.shape)
w, h, _ = faces_unknown[0].shape

faces_fred = np.array([cv.resize(img,(h, w)) for img in np.load(FACES_PATH + '/' + "fred.npy")])
faces_fred_zoomed_out = np.array([cv.resize(img,(h, w)) for img in np.load(FACES_PATH + '/' + "fred_zoomed_out.npy")])
faces_shaggy = np.array([cv.resize(img,(h, w)) for img in np.load(FACES_PATH + '/' + "shaggy.npy")])
faces_shaggy_zoomed_out = np.array([cv.resize(img,(h, w)) for img in np.load(FACES_PATH + '/' + "shaggy_zoomed_out.npy")])
faces_daphne = np.array([cv.resize(img,(h, w)) for img in np.load(FACES_PATH + '/' + "daphne.npy")])
faces_daphne_zoomed_out = np.array([cv.resize(img,(h, w)) for img in np.load(FACES_PATH + '/' + "daphne_zoomed_out.npy")])

faces_velma = np.array([cv.resize(img,(h, w)) for img in np.load(FACES_PATH + '/' + "velma.npy")])
faces_velma_zoomed_out = np.array([cv.resize(img,(h, w)) for img in np.load(FACES_PATH + '/' + "velma_zoomed_out.npy")])

background = np.array([img for img in np.load(BACKGROUND_PATH + "unknown_background.npy")])
faces_flip = np.flip(faces_unknown, axis=2)
faces_fred_flip = np.flip(faces_fred, axis=2)
faces_shaggy_flip = np.flip(faces_shaggy, axis=2)
faces_daphne_flip = np.flip(faces_daphne, axis=2)
faces_velma_flip = np.flip(faces_velma, axis=2)

X = np.concatenate((faces_unknown, faces_unknown_zoomed_out, faces_daphne, faces_flip, faces_fred_flip, faces_shaggy_flip, faces_daphne_flip, faces_velma_flip, faces_fred, faces_shaggy, faces_velma, background, faces_fred_zoomed_out, faces_daphne_zoomed_out, faces_velma_zoomed_out, faces_shaggy_zoomed_out))
y = np.concatenate(
    (np.array([1 for _ in range(len(faces_unknown))]),
     np.array([1 for _ in range(len(faces_unknown_zoomed_out))]),
     np.array([1 for _ in range(len(faces_daphne))]),
     np.ones(faces_flip.shape[0]),
     np.ones(faces_fred_flip.shape[0]),
     np.ones(faces_shaggy_flip.shape[0]),
     np.ones(faces_daphne_flip.shape[0]),
     np.ones(faces_velma_flip.shape[0]),
     np.array([1 for _ in range(len(faces_fred))]),
     np.array([1 for _ in range(len(faces_shaggy))]),
     np.array([1 for _ in range(len(faces_velma))]),
     np.array([0 for _ in range(len(background))]),
     np.array([1 for _ in range(len(faces_velma_zoomed_out))]),
     np.array([1 for _ in range(len(faces_shaggy_zoomed_out))]),
     np.array([1 for _ in range(len(faces_daphne_zoomed_out))]),
     np.array([1 for _ in range(len(faces_fred_zoomed_out))]),
     )
    )
print(X.shape)
print((y == 1).sum())
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.05, stratify=y)

idx = np.nonzero(y_train)


x_train = torch.stack([transforms.ToTensor()(x) for x in x_train])
x_test = torch.stack([transforms.ToTensor()(x) for x in x_test])

train_dataset = TensorDataset(torch.tensor(x_train, dtype= torch.float32),torch.tensor(y_train, dtype= torch.float32))
train_data_loader = DataLoader(train_dataset,batch_size= 128 , shuffle= True)

test_dataset = TensorDataset(torch.tensor(x_test, dtype= torch.float32),torch.tensor(y_test, dtype= torch.float32))
test_data_loader = DataLoader(test_dataset,batch_size= 128 , shuffle= False)

lr = 0.00001

model = nn.Sequential(
        nn.Conv2d(3, 32, (3, 3), padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride= 2),
        nn.Conv2d(32, 64, (3, 3), padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride= 2),
        nn.Conv2d(64, 128, (3, 3), padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride= 2),
        nn.Conv2d(128, 256, (3, 3), padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride= 2),
        nn.Conv2d(256, 512, (3, 3), padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)), 
        nn.Flatten(),
        nn.Linear(512, 1)
)
device = torch.device("mps")
optimizer = optim.Adam(model.parameters(), lr=lr)
weight = (train_dataset.tensors[1] == 0).to(dtype=torch.int32).sum() / (train_dataset.tensors[1] == 1).to(dtype=torch.int32).sum()
criterion = nn.BCEWithLogitsLoss(pos_weight=weight.to(device))
f1_metric_train = F1Score(task="binary").to(device)
f1_metric_val = F1Score(task="binary").to(device)

model.to(device)

for epoch in tqdm(range(35)):
    model.train()
    train_loss = 0.0
    val_loss = 0.0
    f1_metric_train.reset()
    f1_metric_val.reset()
    for x_batch, y_batch in train_data_loader:
        x_batch, y_batch = x_batch.to(device) , y_batch.to(device)
        optimizer.zero_grad()
        output = model(x_batch).squeeze()
        
        if output.shape == torch.Size([]):
            output = output.unsqueeze(0)

        #loss = criterion(output, y_batch)
        loss = sigmoid_focal_loss(output, y_batch, reduction="mean")
        train_loss += loss.item() * x_batch.shape[0]
        f1_metric_train.update(torch.sigmoid(output), y_batch)     
        loss.backward()
        optimizer.step()
    
    model.eval()
    avg_train_loss = train_loss / len(train_data_loader.dataset)
    avg_train_f1 = f1_metric_train.compute()
    
    with torch.no_grad():
        for x_batch, y_batch in test_data_loader:
            x_batch, y_batch = x_batch.to(device) , y_batch.to(device)
            output = model(x_batch).squeeze()
            # loss = criterion(output, y_batch)
            loss = sigmoid_focal_loss(output, y_batch, reduction="mean")
            f1_metric_val.update(torch.sigmoid(output), y_batch)   
            val_loss += loss.item() * x_batch.shape[0]
        
        
        avg_val_loss = val_loss / len(test_data_loader.dataset)
        avg_val_f1 = f1_metric_val.compute()
        
        print(f"Epoch [{epoch+1}/35]")
        print(f"Train - Loss: {avg_train_loss:.4f}, F1: {avg_train_f1:.4f}")
        print(f"Val   - Loss: {avg_val_loss:.4f},   F1: {avg_val_f1:.4f}")
        print("-" * 30)
        
        if avg_val_f1>max_f1:
            max_f1 = avg_val_f1
            torch.save(model, f"./Models/fred/CNNunknown5sigmoid_flip.pth")