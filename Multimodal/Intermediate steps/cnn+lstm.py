import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
from PIL import Image
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, f1_score, recall_score

class Flickr30kDataset(Dataset):
    def __init__(self, captions_file, image_dir, transform=None):
        print("Initializing dataset...")
        self.captions = pd.read_csv(captions_file, sep='|', names=["image", "comment_number", "comment"])
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print("Dataset initialized with {} samples.".format(len(self.captions)))
    
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        image_name = self.captions.iloc[idx, 0].strip()
        image_path = os.path.join(self.image_dir, image_name)
        
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        caption = self.captions.iloc[idx, 2].strip()
        tokens = self.tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=30)
        
        caption_length = min(len(caption.split()), 30) 
        return image, tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze(), caption_length

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(ImageCaptioningModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  
        self.resnet.eval()  
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.embed = nn.Embedding(vocab_size, embed_size)
    
    def forward(self, images, captions, lengths):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        hiddens, _ = self.lstm(packed)
        
        outputs = self.linear(hiddens[0])
        return outputs

def evaluate_model(model, dataloader, device):
    print("Evaluating model...")
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for images, captions, _ in dataloader:
            images = images.to(device)
            captions = captions.to(device)
            
            outputs = model(images, captions, lengths=[len(c) for c in captions])
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(captions.view(-1).cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test Recall: {recall:.4f}")
    return accuracy, f1, recall

def main():
    captions_file = '/Users/dariadragomir/AI_siemens/Flickr30k/dataset/results.csv'
    image_dir = '/Users/dariadragomir/AI_siemens/Flickr30k/dataset/flickr30k_images'
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("Loading dataset...")
    dataset = Flickr30kDataset(captions_file, image_dir, transform)
    
    train_size = int(0.15 * len(dataset))
    test_size = int(0.05 * len(dataset))
    unused_size = len(dataset) - train_size - test_size

    train_dataset, test_dataset, _ = random_split(dataset, [train_size, test_size, unused_size])
    print(f"Dataset split into {train_size} training and {test_size} testing samples.")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    embed_size = 256
    hidden_size = 512
    vocab_size = len(BertTokenizer.from_pretrained('bert-base-uncased').vocab)
    num_layers = 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Initializing model...")
    model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (images, captions, _, lengths) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)
            

            outputs = model(images, captions, lengths)
            loss = criterion(outputs, captions.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            total_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {total_loss / len(train_loader):.4f}")
    

    print("Evaluating model on test data...")
    evaluate_model(model, test_loader, device)
    print("Training and evaluation complete.")

if __name__ == "__main__":
    main()
