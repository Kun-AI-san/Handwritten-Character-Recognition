"""
Image Character Classification using Deep Learning

"""

import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
#training, validation and test data and their respective label locations
train_data_file = 'kmnist-train-imgs.npz' 
train_data_labels_file = 'kmnist-train-labels.npz'
validation_data_file = 'kmnist-val-imgs.npz'
validation_data_labels_file = 'kmnist-val-labels.npz'
test_data_file = 'kmnist-test-imgs.npz' 
test_data_labels_file = 'kmnist-test-labels.npz'


#training data transform composition sequence
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0.5, 0.5),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.Pad(4),
    torchvision.transforms.RandomCrop(28)
    ])

#validation and test data transformation sequence
val_test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(0.5, 0.5)
    ])

#define a custom dataset class
class ClassifierDataset(Dataset):
    def __init__(self, data_file, label_file, transform):
        #load the datasets
        self.data = np.load(data_file)['arr_0']
        self.labels = np.load(label_file)['arr_0']
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        #transform the datasets with their respective transformations mentioned
        image = self.transform(self.data[index])
        label = torch.tensor(self.labels[index])
        return image, label
#create training, validation and test set datasets using the custom dataset class.
training_set = ClassifierDataset(train_data_file, train_data_labels_file, train_transform)
val_set = ClassifierDataset(validation_data_file, validation_data_labels_file, val_test_transform)
test_set = ClassifierDataset(test_data_file, test_data_labels_file, val_test_transform)

#create a Deep learning pipeline 
class NeuralNet(nn.Module):
    def __init__(self, filter_sizes, strides, paddings, out_channels,  in_channels=1, output_dim=10):
        super(NeuralNet,self).__init__()
        """
        Inputs:
        nn.Sequential layer 1: 1st layer of covolution,
        input images dimension: [batch_size, 1, 28,28]
        input channel: 1
        output channels: 32
        stride: 1
        padding: 2
        filter size: 5
        activation: ReLU
        maxpool: kernel:2, stride:2
        Outputs: output image of dimension [batch_size, 32, 14, 14]
        """
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels[0], 
                                             kernel_size=filter_sizes[0], stride=strides[0], padding=paddings[0]),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2)
                                   )
        """
        Inputs:
        nn.Sequential layer 1: 2st layer of covolution,
        input images dimension: [batch_size, 32, 14, 14]
        input channel: 32
        output channels: 64
        stride: 1
        padding: 1
        filter size: 3
        activation: ReLU
        maxpool: kernel:2, stride:2
        Outputs: output image of dimension [batch_size, 64, 7, 7]
        """
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[1], 
                                             kernel_size=filter_sizes[1], stride=strides[1], padding=paddings[1]),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2)
                                   )
        """
        Fully connected layer with 7*7*64 inputs and 1024 outputs
        """
        self.fc1 = nn.Linear(7*7*64, 1024)
        """
        Fully connected layer with 1024 inputs and 10 outputs
        """
        self.fc2 = nn.Linear(1024, 10)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        #flattening the output to make suitable for fully connected layer
        out = out.reshape((out.shape[0], out.shape[1]*out.shape[2]*out.shape[3]))
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

#define epoch and batch size
epochs = 30
batch_size= 128#64#256   

#define training, validation and test data loaders
train_loader = DataLoader(training_set,batch_size=batch_size)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)
#define all the hyperparameters
filter_sizes = [5, 3]
channels = [32, 64]
strides = [1,1]
paddings = [2,1]
#Initialize the CNN network built earlier
conv_net = NeuralNet(filter_sizes, strides, paddings, channels)
#move the CNN network to GPU if GPU is available
conv_net.to(dev)
#initialize Cross Entropy Loss
losses = nn.CrossEntropyLoss()
#initialize the Adam optimizer with given specifications
optimizer = optim.Adam(conv_net.parameters(),lr = 1e-3, betas=(0.9,0.999))
train_losses = []
train_preds = []
val_losses = []
train_accuracy = []
val_accuracy = []
for i in range(epochs):
    score = 0.0
    loss = 0.0
    size = 0
    n = 0
    # Train model
    for j, data in enumerate(train_loader, 0):
        X, Y = data
        X = X.type(torch.float).to(dev)
        Y = Y.type(torch.long).to(dev)
        predictions = conv_net(X)
        optimizer.zero_grad()
        Losses = losses(predictions, Y)
        _, y = torch.max(predictions, 1)
        score+=(y==Y).sum()
        optimizer.zero_grad()
        Losses.backward()
        optimizer.step()
        loss += Losses.item()
        size+=X.shape[0]
        n+=1
    #Compute the training loss and accuracy
    loss/=n
    score1 = score.detach().cpu().numpy()
    score1/=size
    train_accuracy.append(score1*100)
    train_losses.append(loss)
    #PATH = './checkpoints/CNN_{:02d}.pth'.format(i)
    #torch.save(conv_net.state_dict(), PATH)
    score = 0.0
    loss = 0.0
    size=0
    n = 0
    #Validating the model
    for j, data in enumerate(val_loader, 0):
        with torch.no_grad():
            X, Y = data
            X = X.type(torch.float).to(dev)
            Y = Y.type(torch.long).to(dev)
            predictions = conv_net(X)
            Losses = losses(predictions, Y)
            _, y = torch.max(predictions, 1)
            score+=(y==Y).sum()
            loss += Losses.item()
            size+=X.shape[0]
            n+=1
    #Compute the validation loss and accuracy
    loss/=n
    score1 = score.detach().cpu().numpy()
    score1/=size
    val_accuracy.append(100*score1)
    val_losses.append(loss)
score = 0.0   
index=0
#classify test set
for i, data in enumerate(test_loader, 0):
    with torch.no_grad():
        X, Y = data
        index+=batch_size
        X = X.type(torch.float).to(dev)
        Y = Y.type(torch.long).to(dev)
        predictions = conv_net(X)
        _, y = torch.max(predictions, 1)
        score+=(y==Y).sum()
        n+=1
#Compute the Test Accuracy
score = score.detach().cpu().numpy()
score/=index
plt.plot(train_losses)
plt.xlabel("Number of epochs")
plt.ylabel("Training Loss")
plt.show()
plt.plot(val_losses)
plt.xlabel("Number of epochs")
plt.ylabel("Valdation Loss")
plt.show()
plt.plot(train_accuracy)
plt.xlabel("Number of epochs")
plt.ylabel("Training Accuracy")
plt.show()
plt.plot(val_accuracy)
plt.xlabel("Number of epochs")
plt.ylabel("Validation Accuracy")
plt.show()
print("The Test set Accuracy (%): ", score*100)
    
        
    
    
        
            

