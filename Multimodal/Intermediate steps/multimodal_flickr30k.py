import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class Flickr30kDataset(Dataset):
    def __init__(self, image_dir, captions_file, transform=None, tokenizer=None, max_length=30):
        self.image_dir = image_dir
        self.captions = []
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(captions_file, 'r') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 3:
                    img_name = parts[0].strip()
                    caption = parts[2].strip()
                    self.captions.append((img_name, caption))
    
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        img_name, caption = self.captions[idx]
        image_path = os.path.join(self.image_dir, img_name)
        
        image = None
        
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"File not found: {image_path}") 
            return None, None, None  

        if self.transform:
            image = self.transform(image)

        encoding = self.tokenizer(
            caption, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return image, input_ids, attention_mask

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

image_dir = "/Users/dariadragomir/AI_siemens/Flickr30k/dataset/flickr30k_images"
captions_file = "/Users/dariadragomir/AI_siemens/Flickr30k/dataset/results.csv"
dataset = Flickr30kDataset(image_dir, captions_file, transform=image_transform, tokenizer=tokenizer)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class MultimodalNet(nn.Module):
    def __init__(self):
        super(MultimodalNet, self).__init__()
        self.vision_model = models.resnet50(pretrained=True)
        self.vision_model.fc = nn.Identity() 
        
        self.nlp_model = BertModel.from_pretrained("bert-base-uncased")
        
        vision_output_size = 2048
        nlp_output_size = 768
        self.fc = nn.Linear(vision_output_size + nlp_output_size, 256)
        self.classifier = nn.Linear(256, 2)  

    def forward(self, image, input_ids, attention_mask):
        vision_features = self.vision_model(image)
        nlp_outputs = self.nlp_model(input_ids=input_ids, attention_mask=attention_mask)
        nlp_features = nlp_outputs.pooler_output
        
        combined_features = torch.cat((vision_features, nlp_features), dim=1)
        
        x = self.fc(combined_features)
        x = torch.relu(x)
        output = self.classifier(x)
        
        return output

def train(model, dataloader, epochs=5):
    print("ceva")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, input_ids, attention_mask in dataloader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, input_ids, attention_mask)
            
            labels = torch.ones(images.size(0), dtype=torch.long).to(device) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalNet().to(device)

train(model, train_dataloader)

def evaluate(model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad(): 
        for images, input_ids, attention_mask in dataloader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(images, input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)

            labels = torch.ones(images.size(0), dtype=torch.long).to(device) 
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f'Accuracy: {accuracy:.4f}')

evaluate(model, test_dataloader)
