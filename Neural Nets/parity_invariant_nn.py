#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 14:59:22 2020

@author: vr308

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pylab as plt
import torch.optim as optim
import configparser
import os

x = torch.tensor([[1,2,3],
                  [4,5,6],
                  [7,8,9]])
print(x)
np.pad(x, ((1, #top wrap pad
            1), #bot wrap pad
           (0, #left wrap pad
            0)), #right wrap pad
        mode='wrap')

def my_rotate_180(x):
  return torch.flip(x,[2,3])
  
def my_mirror_TB(x):
  return torch.flip(x,[2])

def my_mirror_LR(x):
  return torch.flip(x,[3])

def my_roll_down(x,distance):
  return torch.roll(x,distance,2)

def my_pad_TB(x, n):
  import torch.nn.functional as F
  return F.pad(input=x, pad=(0, #off the left
                             0, #off the right
                             n, #off the top
                             n), #off the bottom
                mode='circular')
  
def my_pad_bottom(x, n):
  import torch.nn.functional as F
  return F.pad(input=x, pad=(0, #off the left
                             0, #off the right
                             0, #off the top
                             n), #off the bottom
                mode='circular')

def show_some_images(images,max_width=8, max_height=8):
  plt.figure()
  width=max_width
  height=max_height
  batch_size = len(images)
  while height*width>batch_size:
    if height>width:
      height -= 1
    else:
      width -= 1
  num_of_images = height*width
  for index in range(1, num_of_images + 1):
    plt.subplot(width,height, index)
    plt.axis('off')
    plt.imshow(images[index-1].numpy().squeeze(), cmap='gray_r')
    
def float_tensor(X): return torch.tensor(X).float()

class DistortedMNIST(torch.nn.Dataset):
    def __init__(self, test_data):
        
        rolled_down_images = torch.roll(test_data.data[:,None,:,:],1,2)
        reflected_images = torch.flip(test_data.data[:,None,:,:], [3])
        self.jointset_images = float_tensor(torch.vstack((
                                rolled_down_images, reflected_images)))
        self.targets =  torch.cat((test_data.targets,test_data.targets))
        
    def __getitem__(self, index):
        label = self.targets[index] 
        img = self.jointset_images[index]
        return (img, label)

    def __len__(self):
        return len(self.jointset_images)# of how many examples(images?) you have
    

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      
      # Properties of the Image
      self.rowsCols=torch.tensor([28,28])
      self.kernel1_tuple=(4,4) # rows,cols
      self.kernel2_tuple=(28,6) # rows,cols
      self.kernel1_tensor=torch.tensor(self.kernel1_tuple)
      self.kernel2_tensor=torch.tensor(self.kernel2_tuple)

      # First 2D convolutional layer, taking in 1 input channel (B&W image),
      # outputting 32 convolutional features (output channels or colours, etc),
      # with a square kernel size of 3 and a stride of 1.
    
      self.conv1 = nn.Conv2d(1, 32, self.kernel1_tuple, 1)
      # Second 2D convolutional layer, taking in the 32 input layers,
      # outputting 64 convolutional features, with a square kernel size of 3
      self.conv2 = nn.Conv2d(32, 64, self.kernel2_tuple, 1)


      # Designed to ensure that adjacent pixels are either all 0s or all active
      # with an input probability
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)

      self.fc1 = nn.Linear(640, 128)  # for kernel size 4
      # Second fully connected layer that outputs our 10 labels
      self.fc2 = nn.Linear(128, 1)

      #self.conv1 = nn.Conv2d(1, 32, 3, 1)
      #self.conv2 = nn.Conv2d(32, 64, 3, 1)
      #self.dropout2 = nn.Dropout2d(0.5)
      #self.fc1 = nn.Linear(9216, 128)
      #self.dropout1 = nn.Dropout2d(0.25)
      #self.fc2 = nn.Linear(128, 10)

    # x represents our data
    def forward(self, x):
      answer = self.forward_worker(x)
      answer = answer + self.forward_worker(my_rotate_180(x))
      answer = answer - self.forward_worker(my_mirror_TB(x))
      answer = answer - self.forward_worker(my_mirror_LR(x))
      return F.softsign(answer)


    # x represents our data
    def forward_worker(self, x):
      # Note that I am only intending here to flip 28x28 images, but it seems that we are
      # given rank 4 tensors even if there is only one image.  This seems to be (at least in part)
      # so that we can be given (a) multiple images, and (b) each image can have different channels (e.g. r.g.b).
      # It seems that the multi-imaging is for batches, while the multi-channeling is for colours.

      #MMM rowsCols = self.rowsCols
      #MMM print("1. Initial dims should be N, 1,",rowsCols,": ", x.size())
      # Pad data top and bottom so that can treat phi periodically.
      # Note that adding things top and bottom increase the ROWS, and 
      # so the padding size is related to the number of ROWS in first_2d_kernel
      x = my_pad_bottom(x, self.kernel1_tuple[0]-1)  # 0=rows
      #MMM rowsCols += torch.tensor([self.kernel1_tuple[0]-1, 0])
      #MMM print("2. Should have grown to   N,32,",rowsCols,": ", x.size())
      # Pass x data through conv1
      x = self.conv1(x)
      #MMM rowsCols += torch.tensor((1,1)) - self.kernel1_tensor
      #MMM print("3. Should be back at      N,32,",rowsCols,": ", x.size())
      # Use the rectified-linear activation function over x
      x = F.relu(x)

      # Pad data top and bottom so that can treat phi periodically.
      # Note that adding things top and bottom increase the ROWS, and 
      # so the padding size is related to the number of ROWS in second_2d_kernel
      x = my_pad_bottom(x, self.kernel2_tuple[0]-1)  # 0=rows
      #MMM rowsCols += torch.tensor([self.kernel2_tuple[0]-1, 0])
      #MMM print("4. Should have grown to   N,32,",rowsCols,": ", x.size())
      # Pass x data through conv2
      x = self.conv2(x)
      #MMM rowsCols += torch.tensor((1,1)) - self.kernel2_tensor
      #MMM print("5. Should be back at      N,64,",rowsCols,": ", x.size())
      # Use the rectified-linear activation function over x
      x = F.relu(x)
      #MMM print("")
      #MMM 

      # Run max pooling over all phi and some eta
      x = F.max_pool2d(x, (28, 2)) 
      # Pass data through dropout1
      #x = self.dropout1(x)
      # Flatten x with start_dim=1
      x = torch.flatten(x, 1)
      # Pass data through fc1
      #MMM print("5. is now: ", x.size())
      #MMM 
      x = self.fc1(x)
      x = F.relu(x)
      #x = self.dropout2(x)
      x = self.fc2(x)

      return torch.sum(x, dim=1) # because the 128->1 layer gives [1] not 1.

      # Apply softmax to x 
      #output = F.log_softmax(x, dim=1)
      #return torch.einsum("xi,i->x",output,torch.tensor([1.0,-1.0]))
      

# def train(epoch):
#     logging_interval = 10
#     my_nn.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         optimizer.zero_grad()
#         output = network(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % logging_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
#             train_losses.append(loss.item())
#             train_counter.append(
#                 (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
#             torch.save(network.state_dict(), os.getcwd() + '/results/model.pth')
#             torch.save(optimizer.state_dict(), os.getcwd() + '/results/optimizer.pth')
            
# def test():
#       network.eval()
#       test_loss = 0
#       correct = 0
#       with torch.no_grad():
#         for data, target in test_loader:
#           #print('hi')
#           #import pdb; pdb.set_trace();
#           output = network(data)
#           test_loss += F.nll_loss(output, target, size_average=False).item()
#           pred = output.data.max(1, keepdim=True)[1]
#           correct += pred.eq(target.data.view_as(pred)).sum()
#       test_loss /= len(test_loader.dataset)
#       test_losses.append(test_loss)
#       print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))


def how_did_we_do_on(x):
  from math import sqrt
  num_positives=torch.count_nonzero(torch.gt(x,0).int()).item()
  num_negatives=torch.count_nonzero(torch.lt(x,0).int()).item()
  num_zeros=torch.count_nonzero(torch.eq(x,0).int()).item()
  num_total=num_positives+num_zeros+num_negatives
  poisson_p=num_positives/num_total
  poisson_q=1.0-poisson_p
  poisson_mean = num_total*poisson_p
  poisson_variance = num_total*poisson_p*poisson_q
  poisson_sd=sqrt(poisson_variance)
  fractional_mean = poisson_mean/num_total
  fractional_sd = poisson_sd/num_total
  
  if False:
    print("p ",poisson_p)
    print("q ",poisson_q)
    print("np ",poisson_mean)
    print("npq",poisson_variance)
    print("sqrt(npq)",poisson_sd)
    #print(num_positives, num_zeros, num_negatives, num_total)
  
  print("Positive Fraction ",100*fractional_mean," +- ",100*fractional_sd," %")
  #print(torch.transpose(x[:32],0,1))

def how_did_we_do(images):
  how_did_we_do_on(my_nn(images))
  
  
def tint(images,rgb_tints):
    # assume images has dimension (numImages, 1(grey-chan), wid, height)
    return torch.einsum("igxy,ic->icxy",images,rgb_tints)

def images_to_net_tints(images):
    # Assumes net returns values in [-1,1]
    vals_in_0_1 = (my_nn(images)+1)/2;
    right_handed = vals_in_0_1[:,None]
    left_handed  = (1-vals_in_0_1)[:,None]
  
    # Note that the colours given deliberately
    # over-saturate so that the mean colour is yellow.
    # Over-saturation is fine as the image projectors all clip nicely.
    right_handed_colour=torch.tensor([-1,3,0])[None,:] 
    left_handed_colour=torch.tensor([3,-1,0])[None,:]

    return (right_handed*right_handed_colour + 
             left_handed* left_handed_colour)

def images_to_tinted_images(images): 
    # Assumes all images are greyscale.  
    return tint(-images, images_to_net_tints(images))
    
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.detach().numpy()
    #npimg = tint_numpy_image(npimg, [1.0, 0.0, 0.0])
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
        
if __name__ == '__main__':
    
    # Import configurations from external 'config.ini'- that's good practise 
    # as its easier to run with different configs without touching the main src 
    # code
    
    parser = configparser.ConfigParser()
    parser.read('config.ini')
    
    batch_size_train = int(parser['data']['batch_size_train'])
    batch_size_test = int(parser['data']['batch_size_test'])
    learning_rate = float(parser['training_hypers']['learning_rate'])
    momentum = float(parser['training_hypers']['momentum'])
    n_epochs = int(parser['training_hypers']['n_epochs'])
    
    ## 
    train_data =  torchvision.datasets.MNIST('MNIST_train', train=True, download=False,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ]))
    subset_train = torch.utils.data.random_split(train_data, [5000, 55000])
    train_loader = torch.utils.data.DataLoader(subset_train[0], batch_size=batch_size_train, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST( 'MNIST_test', train=False, download=False,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
     batch_size=batch_size_test, shuffle=True)
    
    train_samples = enumerate(train_loader)
    batch_idx, (example_data, example_targets) = next(train_samples)
    
    ##-----------------------------------------------------########

    ## Training block 
    
    ##-----------------------------------------------------########
    
    ## Subsetting to (5000/1000) train/test split to speed up training 
    subset = torch.utils.data.Subset(train_loader, range(0,60000,12))
    sub_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size_train)
    
    my_nn = Net()
    print(my_nn)
    
    print(" Untrained net called with some test data images (note the shape)")
    output = my_nn(example_data)
    
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
    
    # test()
    # for epoch in range(1, n_epochs + 1):
    #     train(epoch)
    #     test()
    
    optimizer = optim.SGD(my_nn.parameters(), lr = learning_rate, momentum=momentum)
    logging_interval = 10
    my_nn.train()
    for epoch in range(3):  # loop over the dataset multiple times
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            custom_loss = -torch.sum(my_nn(data))
            custom_loss.backward()
            optimizer.step()    
            # print statistics at periodic intervals
            if batch_idx % logging_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), custom_loss.item()))
                train_losses.append(custom_loss.item())
                train_counter.append(
                    (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
                torch.save(my_nn.state_dict(), os.getcwd() + '/results/model.pth')
                torch.save(optimizer.state_dict(), os.getcwd() + '/results/optimizer.pth')    
                how_did_we_do(data)
    
    
    print('Finished Training')
    how_did_we_do()
    
    PATH = './my_nn.pth'
    torch.save(my_nn.state_dict(), PATH)
    
    ##-----------------------------------------------------########

    ## Testing block -- how did we do on previously unseen images
    
    ##-----------------------------------------------------########
    
    ##-----------------------------------------------------########
    
    ## Vizualisation block
    
    ##-----------------------------------------------------########
        
    # print images
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images_to_tinted_images(images)))
    imshow(torchvision.utils.make_grid(my_roll_down(images_to_tinted_images(images),10)))
    how_did_we_do()
    #images, labels = dataiter.next()
    #imshow(torchvision.utils.make_grid(images_to_tinted_images(my_rotate_180(images))))
    
    #images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images_to_tinted_images(my_mirror_LR(images))))
    imshow(torchvision.utils.make_grid(images_to_tinted_images(my_roll_down(my_mirror_LR(images),10))))
    #print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(4)))
    
    my_net = Net()
    my_net.load_state_dict(torch.load(PATH))