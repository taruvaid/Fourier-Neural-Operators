# -*- coding: utf-8 -*-
"""mlp_weight_matrices_and_init_schemes.ipynb


"""

print("Weights Concepts".center(100,'-'))
concepts = '''1. Weight matrix dimensions = nodes in current layer x nodes in previous layer\n
2. Models cannot learn when all trainable parameters are initalized to the same value
3. Models can learn as long as some trainable parameters are initialized to different numbers.

'''
print(concepts)

import torch
import torch.nn as nn

print("Model 1".center(100,'-'))
model_1 = nn.Sequential(
    nn.Linear(10,14), #input layer
    nn.Linear(14,19), #hidden layer
    nn.Linear(19,8) #output layer
)

print(model_1)

for i in range(len(model_1)):
  print(model_1[i].weight.shape)

print("Model 2".center(100,'-'))

model_2 = nn.Sequential(
    nn.Linear(10,14), #input layer
    nn.Linear(14,9), #hidden layer
    nn.Linear(19,8) #output layer
)
print(model_2)

for i in range(len(model_2)):
  print(model_2[i].weight.shape)

print("Model 3".center(100,'-'))
model_3 = nn.Sequential(
    nn.Linear(2,3), #input layer
    nn.Linear(3,3), #hidden layer
    nn.Linear(3,1) #output layer
)

print(model_3)

for i in range(len(model_1)):
  print(model_3[i].weight.shape)

nsamples = 5
nfeatures = 10

testdata = torch.randn(nsamples,nfeatures)
testdata.shape

model_1(testdata).shape
#model_2(testdata) #throws an error due to incorrect node sizes

nsamples = 5
nfeatures = 2

testdata = torch.randn(nsamples,nfeatures)
print("/*Input data & its shape*/")
print(testdata.shape)
print(testdata)
W = []
print("\n/*Weights & Biases*/")
for i in range(len(model_3)):
  print(f"Layer {i}")
  for name, params in model_3[i].named_parameters():
    print(f"{name} = {params}")
    W.append(params)

print("\n/*Output*/")
print(model_3(testdata))

print("\n/*Replciating result using matrix multiplication*/")
print(" This is essentially xwT")
for i in range(testdata.shape[0]):
  print(torch.matmul(torch.matmul(torch.matmul(testdata[i], W[0].T) + W[1],W[2].T)+ W[3],W[4].T) + W[5])

testdata.shape[0]

"""#msnist and initializations"""

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

data = np.loadtxt(open("/content/sample_data/mnist_train_small.csv"),delimiter=',')
data.shape #28 x 28 + label

i = 951
print(f"label of image in row no {i} :", int(data[i-1][0]))
plt.imshow(data[i-1][1:].reshape(28,28), cmap='gray')

labels = data[:,0]
data = data[:,1:]

print(f"Pre Normalization : data min value {data.min()}, data max value {data.max()}")

#normalize data
dataNorm = data/data.max()
print(f"Post Normalization : data min value {dataNorm.min()}, data max value {dataNorm.max()}")

#convert to tensor
dataT = torch.tensor(dataNorm).float()
labelsT = torch.tensor(labels).long() # Convert labels to long type (integer)

#train test split
train_data,test_data, train_labels,test_labels = train_test_split(dataT, labelsT, test_size=.1)
print(train_data.shape, train_labels.shape)
print(test_data.shape, test_labels.shape)

#convert into PyTorch Datasets
train_data = torch.utils.data.TensorDataset(train_data,train_labels)
test_data  = torch.utils.data.TensorDataset(test_data,test_labels)

#dataloader objects
batchsize    = 32
train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True,drop_last=True)
test_loader  = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])

#define the model
def mnist_1():

  class mnist_1(nn.Module):
    def __init__(self):
      super().__init__()

      ### input layer
      self.input = nn.Linear(784,64)

      ### hidden layer
      self.fc1 = nn.Linear(64,32)
      self.fc2 = nn.Linear(32,32)

      ### output layer
      self.output = nn.Linear(32,10)

    # forward pass
    def forward(self,x):
      x = F.relu( self.input(x) )
      x = F.relu( self.fc1(x) )
      x = F.relu( self.fc2(x) )
      return self.output(x)

  # create the model instance
  model = mnist_1()

  # loss function
  lossfun = nn.CrossEntropyLoss()

  # optimizer
  optimizer = torch.optim.Adam(model.parameters(),lr=.01)

  return model,lossfun,optimizer

model_1 = mnist_1()[0]
print(model_1)

print('\nWeight shapes of submodules:')
for name, module in model_1.named_children():
  if hasattr(module, 'weight'):
    print(f"  {name}: {module.weight.shape}")

print('\n\nWeights for layer fc1:')
print(model_1.fc1.weight.data)

def train_1(model,lossfun,optimizer):

  # number of epochs
  numepochs = 10

  # initialize losses
  losses    = torch.zeros(numepochs)
  trainAcc  = []
  testAcc   = []


  # loop over epochs
  for epochi in range(numepochs):

    # switch on train mode
    model.train()

    # loop over training data batches
    batchAcc  = []
    batchLoss = []
    for X,y in train_loader:

      # forward pass and loss
      yHat = model(X)
      loss = lossfun(yHat,y)

      # backprop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # loss from this batch
      batchLoss.append(loss.item())

      # compute accuracy
      matches = torch.argmax(yHat,axis=1) == y     # booleans (false/true)
      matchesNumeric = matches.float()             # convert to numbers (0/1)
      accuracyPct = 100*torch.mean(matchesNumeric) # average and x100
      batchAcc.append( accuracyPct )               # add to list of accuracies
    # end of batch loop...

    # now that we've trained through the batches, get their average training accuracy
    trainAcc.append( np.mean(batchAcc) )

    # and get average losses across the batches
    losses[epochi] = np.mean(batchLoss)

    # test accuracy
    model.eval()
    X,y = next(iter(test_loader)) # extract X,y from test dataloader
    with torch.no_grad(): # deactivates autograd
      yHat = model(X)

    # compare the following really long line of code to the training accuracy lines
    testAcc.append( 100*torch.mean((torch.argmax(yHat,axis=1)==y).float()) )
  # end epochs

  # function output
  return trainAcc,testAcc,losses,model

# Run the model without changing the weights; this will be the baseline performance.
# Notice the model creation is outside the training
net_base,lossfun,optimizer = mnist_1()
trainAcc_base,testAcc_base,losses,net_base = train_1(net_base,lossfun,optimizer)

# plot the results
plt.plot(range(len(trainAcc_base)),trainAcc_base,'o-', range(len(testAcc_base)),testAcc_base ,'s-')
plt.legend(['Train','Test'])
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.show()

# Change the weights before training
net_zero,lossfun,optimizer = mnist_1()

# set to zeros
net_zero.fc1.weight.data = torch.zeros_like( net_zero.fc1.weight )

# confirm
net_zero.fc1.weight.data

# run the model and show the results
trainAcc_zero,testAcc_zero,losses,net_zero = train_1(net_zero,lossfun,optimizer)

plt.plot(range(len(trainAcc_base)),trainAcc_base,'b-', range(len(testAcc_base)),testAcc_base ,'b:')
plt.plot(range(len(trainAcc_zero)),trainAcc_zero,'r-', range(len(testAcc_zero)),testAcc_zero ,'r:')
plt.legend(['Train base','Test base','Train fc1=zero','Test fc1=zero'])
plt.title('Accuracy comparison with layer FC1 init to zeros')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.show()

# Are the weights still zeros?
print(net_zero.fc1.weight.data)

# show the distributions in a histogram
y,x = np.histogram(net_base.fc2.weight.data.flatten(),30)
plt.plot((x[1:]+x[:-1])/2,y,'r',label='Baseline')

y,x = np.histogram(net_zero.fc2.weight.data.flatten(),30)
plt.plot((x[1:]+x[:-1])/2,y,'b',label='FC1=zeros')

plt.legend()
plt.xlabel('Weight value')
plt.ylabel('Count')
plt.show()

# Change the weights before training
net_allzero,lossfun,optimizer = mnist_1()

# loop over parameters and set them all to zeros
for p in net_allzero.named_parameters():
  p[1].data = torch.zeros_like( p[1].data )


# and confirm for a few select parameters (y-axis offset for visibility)
plt.plot(0+net_allzero.fc1.weight.data.flatten(),'bo')
plt.plot(1+net_allzero.fc2.weight.data.flatten(),'rx')
plt.plot(2+net_allzero.fc1.bias.data.flatten(),'g^')
plt.xlabel('Parameter index')
plt.ylim([-1,3])
plt.ylabel('Parameter value')
plt.show()

# run the model and show the results
trainAcc_allzero,testAcc_allzero,losses,net_allzero = train_1(net_allzero,lossfun,optimizer)

plt.plot(range(len(trainAcc_base)),trainAcc_base,'b-', range(len(testAcc_base)),testAcc_base ,'b:')
plt.plot(range(len(trainAcc_allzero)),trainAcc_allzero,'r-', range(len(testAcc_allzero)),testAcc_allzero ,'r:')
plt.legend(['Train base','Test base','Train all zero','Test all zero'])
plt.title('Accuracy comparison with all layers init to zeros')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.show()

# show the distributions in a histogram
y,x = np.histogram(net_base.fc1.weight.data.flatten(),30)
plt.plot((x[1:]+x[:-1])/2,y,'r',label='Baseline')

y,x = np.histogram(net_allzero.fc1.weight.data.flatten(),30)
plt.plot((x[1:]+x[:-1])/2,y,'b',label='All zeros')

plt.legend()
plt.xlabel('Weight value')
plt.ylabel('Count')
plt.show()
#baseline model has a better distribution

# woah, not even a single non-zero weight value?!?!!?!!??
plt.plot(net_allzero.fc1.weight.data.flatten(),'o');

for p in net_allone.named_parameters():
  print(p[0])

# Change the weights before training
net_allone,lossfun,optimizer = mnist_1()
for p in net_allone.named_parameters():
  print("p",p[1])
  p[1].data = torch.zeros_like( p[1].data ) + 1
  #p[1].data = torch.zeros( p[1].data.shape ) + 1 # equivalent to the previous line!

# Change the weights before training
net_allone,lossfun,optimizer = mnist_1()
for p in net_allone.named_parameters():
  print("p",p[0],p[1])
  p[1].data = torch.zeros_like( p[1].data ) + 1
  #p[1].data = torch.zeros( p[1].data.shape ) + 1 # equivalent to the previous line!

# run the model and show the results
trainAcc_allone,testAcc_allone,losses,net_allone = train_1(net_allone,lossfun,optimizer)

plt.plot(range(len(trainAcc_base)),trainAcc_base,'b-', range(len(testAcc_base)),testAcc_base ,'b:')
plt.plot(range(len(trainAcc_allone)),trainAcc_allone,'r-', range(len(testAcc_allone)),testAcc_allone ,'r:')
plt.legend(['Train base','Test base','Train all ones','Test all ones'])
plt.title('Accuracy comparison with all layers init to ones')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.show()

# Change the weights before training
net_allone,lossfun,optimizer = mnist_1()
for p in net_allone.named_parameters():
  print("p",p[1])
  p[1].data = torch.zeros_like( p[1].data ) + 5


# run the model and show the results
trainAcc_allone,testAcc_allone,losses,net_allone = train_1(net_allone,lossfun,optimizer)

plt.plot(range(len(trainAcc_base)),trainAcc_base,'b-', range(len(testAcc_base)),testAcc_base ,'b:')
plt.plot(range(len(trainAcc_allone)),trainAcc_allone,'r-', range(len(testAcc_allone)),testAcc_allone ,'r:')
plt.legend(['Train base','Test base','Train all 5','Test all 5'])
plt.title('Accuracy comparison with all layers init to ones')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.show()

"""### Understanding `weight` vs `parameters()`

*   **`model_1[i].weight`**: This directly accesses the `weight` attribute of a specific layer in your `nn.Sequential` model. It returns the `torch.Tensor` representing the weights for that layer.

*   **`model_1[i].parameters()`**: This is a method that returns an *iterator* over all the parameters (including both `weight` and `bias` if `bias=True` for that layer) of a specific layer. It's commonly used when you want to iterate through all trainable parameters for optimization, e.g., when passing them to an optimizer.

Let's see them in action:
"""

# Accessing the weight tensor of the first layer
print("model_1[0].weight:")
print(model_1[0].weight)
print(f"Type of model_1[0].weight: {type(model_1[0].weight)}")
print(f"Shape of model_1[0].weight: {model_1[0].weight.shape}")

# Accessing the parameters (weight and bias) of the first layer
print("model_1[0].parameters():")
for name, param in model_1[0].named_parameters():
    print(f"  Parameter name: {name}, shape: {param.shape}")
    if name == 'weight':
        # Verify that it's the same weight tensor
        assert torch.equal(param, model_1[0].weight)
        print("  (This is the same weight tensor as above)")
    elif name == 'bias':
        print("  (This is the bias tensor)")

"""# Xavier Initialization"""







