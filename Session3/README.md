# Model to generate this network
![alt text](https://github.com/anuda/END2/blob/main/Session3/image.png)

# Data Generation Strategy
- Create Custom Dataset
- Input is going to be MNIST image and a random number (0,9)
- this iterator would return (image, number, sum of these 2)

# Create Network
- 2 Convolution layers
- 4 Linear layers
- Model to have 2 inputs and 2 Outputs
- The 2 inputs are concatenated at the latter layers. Once we get the image into an 1D tensor, the number tensor is concatenated

```Python
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=121,out_features=60)
        self.out = nn.Linear(in_features=60,out_features=20)
        self.out2 = nn.Linear(in_features=120,out_features=10)
        
    def forward(self,t,x):
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t,kernel_size=2, stride=2)
        
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t,kernel_size=2, stride=2)
        
        t = t.reshape(-1,12*4*4)
        
        t = self.fc1(t)
        t = F.relu(t)
        
        out2 = self.out2(t)
        out2 = F.log_softmax(out2)

        t = torch.cat(([t, torch.unsqueeze(x,dim=1)]), 1)

        t = self.fc2(t)
        t = F.relu(t)
        
        t=self.out(t)
        
        return out2,t
```

# Model Parameters:
```Python
conv1.weight  :  torch.Size([6, 1, 5, 5])
conv1.bias  :  torch.Size([6])
conv2.weight  :  torch.Size([12, 6, 5, 5])
conv2.bias  :  torch.Size([12])
fc1.weight  :  torch.Size([120, 192])
fc1.bias  :  torch.Size([120])
fc2.weight  :  torch.Size([60, 121])
fc2.bias  :  torch.Size([60])
out.weight  :  torch.Size([20, 60])
out.bias  :  torch.Size([20])
out2.weight  :  torch.Size([10, 120])
out2.bias  :  torch.Size([10])
```

# Sample Output
![alt text](https://github.com/anuda/END2/blob/main/Session3/output.png)
