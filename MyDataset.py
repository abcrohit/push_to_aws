import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
class Autoencoder1(nn.Module):
    def __init__(self):
        super(Autoencoder1,self).__init__()
        input_size=166
        feature_size=30
        
        self.encoder=nn.Sequential(
            nn.Linear(input_size,120),
            nn.ReLU(),
            nn.Linear(120,60),
            nn.ReLU(),
            nn.Linear(60,feature_size)
        )
        self.decoder=nn.Sequential(
            nn.Linear(feature_size,60),
            nn.ReLU(),
            nn.Linear(60,120),
            nn.ReLU(),
            nn.Linear(120,input_size),
            nn.Sigmoid()
        )
    def forward(self,x):
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return decoded
            
def trainAE(DataLoader):
#     model parameters 
    epochs=15
    model=Autoencoder1()
    criterion=nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters())
# 
    loss_per_iter = []
    loss_per_batch = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(DataLoader):
            # inputs = inputs.to(device)
            # labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs.float())
            loss = criterion(outputs, inputs.float())
            loss.backward()
            optimizer.step()

            # Save loss to plot
            running_loss += loss.item()
            loss_per_iter.append(loss.item())
        print(epoch,loss.item())
        loss_per_batch.append(running_loss / (i + 1))
        running_loss = 0.0
    torch.save(model.state_dict(),'AE.pth')

    
    
class MyDataset(Dataset):
    def __init__(self,df):
        x=df.iloc[:,0:-1].values
        y=df.iloc[:,-1].values.tolist()
        y=[int(i) for i in y]
 
    self.x_train=torch.tensor(x,dtype=torch.float32)
    self.y_train=torch.tensor(y,dtype=torch.float32)
 
    def __len__(self):
        return len(self.y_train)
    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]
    
model=Autoencoder1()
model.load_state_dict(torch.load("AE.pth"))

def transform(x):
    x=torch.tensor(x,dtype=torch.float32)
    return model.encoder(x).detach().numpy()
class MyDatasetCompress(Dataset):
    def __init__(self,df):
        x=transform(df.iloc[:,:-1].values)
        y=df.iloc[:,-1].values.tolist()
        y=[int(i) for i in y]
        self.x_train=torch.tensor(x,dtype=torch.float32)
        self.y_train=torch.tensor(y,dtype=torch.float32)
    def __len__(self):
        return len(self.y_train)
    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]
class datasetfromtensor(Dataset):
    def __init__(self,X,y):
        self.x_train=X
        self.y_train=y
    def __len__(self)
        return len(self.y_train)
    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]
    
