import torch
import torchvision
import torch.nn as nn 
import torchvision.transforms as transforms
import torch.nn.functional as F 
import torch.optim as optim

train_set = torchvision.datasets.FashionMNIST(root="",
train=True,
download=True,
transform=transforms.Compose([transforms.ToTensor()]))

def get_num_correct(prediction,labelss):
    return prediction.argmax(dim=1).eq(labelss).sum()



class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5)
        #dence layer
        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10) 
        
    def forward(self,t):
        #input 
        t=t 
        #hideen conv layer
        t =self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t,kernel_size=2,stride=2)
        
        # con 2nd layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t,kernel_size=2,stride=2)
        
        t= t.reshape(-1,12*4*4)
        t= self.fc1(t)
        t = F.relu(t)
        
        t= self.fc2(t)
        t = F.relu(t)
        
        t = self.out(t)
        #t = F.softmax(t,dim=1)
        
        return t 
    
network = Network()
train_loader = torch.utils.data.DataLoader(
                       train_set,
                       batch_size=100)
optimizer = optim.Adam(network.parameters(),lr=0.01)

total_loss = 0 
total_corrct = 0


for batch in train_loader:
    batch = next(iter(train_loader))
    images , labels = batch 

    preds = network(images)

    loss= F.cross_entropy(preds,labels)
    
    optimizer.zero_grad() #pyptorch accumulate add to currently
    loss.backward() #Gradient
    optimizer.step() #update weight
    
    total_loss += loss.item()
    total_corrct += get_num_correct(preds,labels)
    
print(f'EPOCH {0} total_correct {total_corrct} loss {total_loss}')
    