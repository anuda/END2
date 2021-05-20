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
