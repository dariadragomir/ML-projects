import torch.nn as nn
import cv2 as cv
import glob
import pandas as pd
import torchvision.transforms.v2
from tqdm import tqdm
import numpy as np
import torch
from params import *
from matplotlib import pyplot as plt
import torchvision
import random
from torchvision.transforms import functional as F

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# transformare random care roteste imaginea cu un unghi din lista[0, 90, 180, 270]
class RandomSelectRotation:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return F.rotate(img, angle)

# citesc etichetele din csv si le sortez dupa id
def import_labels(path):
    df = pd.read_csv(path, sep=",")
    df = df.sort_values(by="image_id")
    
    return np.array(df["label"].values.astype(np.int64))

# citesc imaginile si le normalizez la [0,1] prin /255
def import_images(path, color):
    files = sorted(glob.glob(path))
    return np.array([cv.imread(files[i], cv.IMREAD_COLOR if color else cv.IMREAD_GRAYSCALE) for i in tqdm(range(len(files)), desc="Importing images")], dtype=np.float32) / 255.0

class DeepFakeDataset(torch.utils.data.Dataset):
    def __init__(self, path, labels_path, transform=None):
        self.images = import_images(path, color=True)
        self.labels = torch.from_numpy(import_labels(labels_path))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
if __name__ == "__main__":
   
    # transformari incercate pt antrenare 
    transforms_train = torchvision.transforms.Compose([
        #torchvision.transforms.ToPILImage(),
        #torchvision.transforms.RandomResizedCrop(100, scale=(0.8, 1.0)),
        #torchvision.transforms.RandomHorizontalFlip(),
        #torchvision.transforms.RandomApply([
        #    torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
        # ], p=0.5),
        #RandomSelectRotation([0, 90, 180, 270]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.414, 0.462, 0.475], std=[0.257, 0.252, 0.255])
    ])

    transforms_val = torchvision.transforms.Compose([
        #torchvision.transforms.ToPILImage(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.41468218,  0.46208042, 0.4756669], std=[0.25712234, 0.2529903,  0.25524536]),
    ])
    
    # creez dataseturile si dataloaderele
    train_dataset = DeepFakeDataset(path="deepfake-classification-unibuc/train/*.png", labels_path="deepfake-classification-unibuc/train.csv", transform=transforms_train)
    val_dataset = DeepFakeDataset(path="deepfake-classification-unibuc/validation/*.png", labels_path="deepfake-classification-unibuc/validation.csv", transform=transforms_val)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # CNN cu 4 blocuri convolutionale (cu batchnorm si relu), 2 max pooling, 2 fully connected si clasificator(2 straturi fully-connected + dropout)
    model = nn.Sequential(
        nn.Conv2d(3,32,3,padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32,32,3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2,2),
        nn.Conv2d(32,64,3,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64,64,3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2,2),
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Dropout(0.1),
        nn.Linear(64, 32),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Linear(32, NUM_CLASSES)
    )
    
    device = DEVICE
    model.to(device)
    
    # functia de loss cross entorpy cu label smoothing 
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # optimizator Adam cu lr=1e-3 si 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    
    # curba de cosinus pentru a reduce lr de la valoarea 1e-3 la 1e-5 in 160 de epoci
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LEARNING_RATE/100)
    
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    
    #mixup = torchvision.transforms.v2.MixUp(num_classes=NUM_CLASSES, alpha=0.4)
    
    # loop de antrenare
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        # total si correct sunt folosite pentru a calcula acuratetea
        correct = 0
        total = 0 

        for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            # aplic mixup daca e activat
            # zero_grad() este apelat pentru a reseta gradientii inainte de fiecare pas de antrenare
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()   # calcularea gradientilor si actualizarea parametrilor
            optimizer.step()

            # running_loss este suma pierderilor pentru a calcula media la finalul epocii
            # predicted este clasa prezisa
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # actualizez rata de invatare
        scheduler.step()

        print(f"Train Loss: {running_loss/len(train_dataloader):.4f}, Train Accuracy: {100 * correct / total:.2f}%")
        train_acc.append(100 * correct / total)
        train_loss.append(running_loss/len(train_dataloader))
        model.eval() # trec la modul evaluare pentru a dezactiva dropout si batch normalization pe batch-ul curent
        
        # la finalul fiecarei epoci evaluez modelul pe setul de validare
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad(): # dezactivez gradientii
            for images, labels in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1) # predicted are clasa cu cea mai mare probabilitate
                total += labels.size(0)
                correct += (predicted == labels).sum().item() 

            print(f"Val Loss: {running_loss/len(val_dataloader):.4f}, Val Accuracy: {100 * correct / total:.2f}%")
        
            val_acc.append(100 * correct / total)
            val_loss.append(running_loss/len(val_dataloader))
            
            if val_acc[-1] > best_acc:
                best_acc = val_acc[-1]

            model.eval()
            if(100*correct / total > 95):
                torch.save(model, f"models/cnn_epoch_{epoch+1}_{100*correct / total}.pth")
                # print(f"Model saved at epoch {epoch+1} with accuracy {100 * correct / total:.2f}%")
    
    # afisez curbele de antrenare si validare
    x = range(1, NUM_EPOCHS + 1)
    plt.plot(x, train_acc, label='Train Accuracy', color='blue')
    plt.plot(x, val_acc, label='Validation Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()
    plt.show()
    plt.plot(x, train_loss, label='Train Loss', color='blue')
    plt.plot(x, val_loss, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.show()
