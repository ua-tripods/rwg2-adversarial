#### mnist_train.py ####
import numpy as np 
import torch
from PIL import Image
from torch.autograd import Variable
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torch.optim as optim 
from torchvision import transforms
import matplotlib.pyplot as plt
import os

class Net(nn.Module):
    def __init__(self, conf):
        super(Net, self).__init__()
		# define the layers of the network -- place any
                #        network you want here
        pairs = zip(conf[:-1], conf[1:])
        self.fcs = nn.ModuleList(nn.Linear(this_layer, next_layer)
                       for (this_layer, next_layer) in pairs)
        #self.fc1 = nn.Linear(conf[0]**2, conf[1]) # size of the image vector
        #self.fc2 = nn.Linear(  conf[1], conf[2]) # reduce the data several
                                         # times 
        #self.fc3 = nn.Linear(  conf[2], 98)  # ... 
        #self.fc4 = nn.Linear(   98, 10)  # final output must match the
					 # desired number of classes.  

    def forward(self, x):
		# define the activation filters that connect layers
        for layer in self.fcs[:-1]:
          x = F.relu(layer(x))
        x = self.fcs[-1](x)
        return x

# class Net1(nn.Module):
#     def __init__(self):
#         super(Net1, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# net1 = Net1()
# crit = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net1.parameters(), lr=0.001, momentum=0.9)

sf = 2
sw = 28
sl = 28

conf = [int(sl*sw/sf/sf), int(100/sf/sf), 10]
net = Net(conf)
output_fil = nn.CrossEntropyLoss() # error function 
		# pick an optimizer -- pytorch has many
        # TODO : Look up why momentum works
weight_opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9,
                       weight_decay=0.00001)  
		# dataloaders copied from github.com/akshaychawla
def flat_trans(x):
    x.resize_(28*28)
    return x
mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(flat_trans)])
sfile = "C:/Users/Nexus/Google Drive/dropbox/Dropbox/UOFA/0-research/network/rwg2-adversarial/ha/imnet_fgsm.py"
if (os.path.isfile(sfile)):
  ddir = "C:/Users/Nexus/Desktop/Adversarial-Examples-in-PyTorch2/mnist_scale"
  odir = "C:/Users/Nexus/Google Drive/dropbox/Dropbox/UOFA/0-research/network/imnet_examples_nexus"
  ldir = "C:/Users/Nexus/Desktop/Adversarial-Examples-in-PyTorch2/ilsvrc12_imnet_labels"
else:  
  ddir = "/home/bwbell/Desktop/Adversarial-Examples-in-PyTorch/mnist_scale"
  mdir = "/home/bwbell/Desktop/Adversarial-Examples-in-PyTorch/mnist2"
  odir = "/home/bwbell/Dropbox/UOFA/0-research/network/imnet_examples"
		# directory for labels in ilsvrc_12
  ldir = "/home/bwbell/Desktop/Adversarial-Examples-in-PyTorch/ilsvrc12_imnet_labels"
		# list of files for review
ddirs = os.listdir(ddir)
ddirs.sort()
imsize = (int(sw/sf), int(sl/sf))
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
toten = transforms.ToTensor()
traindata = torchvision.datasets.MNIST(root=mdir, train=True,
                                       download=True,
                                       transform=mnist_transform) 
trainloader = torch.utils.data.DataLoader(traindata, batch_size=256,
                                          shuffle=True, num_workers=2) 
testdata  = torchvision.datasets.MNIST(root=mdir, train=False,
                                       download=True,
                                       transform=mnist_transform) 
testloader = torch.utils.data.DataLoader(testdata, batch_size=256,
                                         shuffle=True, num_workers=2) 

for epoch in range(21):
  closs = 0.0 
  print("Starting epoch: {}".format(epoch))
  for i, data in enumerate(trainloader, 0):
    print("starting iteration: {}/{}".format(i, len(trainloader)))
    ip, il = data
    ip, il = Variable(ip), Variable(il)
    # convert ip to new scale
    images = np.array(ip)
    sf = 2
    sl = 28
    sw = 28
    # Reshape
    ims2 = images.reshape(len(ip),sl,sw) #[im.numpy().reshape(sl,sw) for im in images]
    # convert to Image
    ims3 = [Image.fromarray(im) for im in ims2]
    # rescale to sfactor
    ims4 = [im.resize((int(sw/sf),int(sl/sf)), Image.NEAREST) for im in ims3]
    # return to numpy formta
    ims5 = [np.array(im.getdata()).reshape(im.size[0], im.size[1]) for im in ims4]
    #np.array(list(ims4[0].getdata())).reshape([sf*sw,sf*sl])

    #plt.imshow(ims5[0])
    #plt.show()
    ims6 = [im.reshape(int(sl*sw/sf/sf)) for im in ims5]
    ims7 = np.asarray(ims6)
    ipo = torch.tensor(ims7).float()

    weight_opt.zero_grad()
    out = net(ipo)
    loss = output_fil(out, il)
    loss.backward()
    weight_opt.step()
    closs += loss.data[0]
    print('Epoch: {}'.format(epoch))


print("Training Complete")

# TEST 
correct = 0.0 
total = 0 
for data in testloader:
    ip, labels = data
    images = np.array(ip)
    sf = 2
    sl = 28
    sw = 28
    # Reshape
    ims2 = images.reshape(len(ip),sl,sw) 
    # convert to Image
    ims3 = [Image.fromarray(im) for im in ims2]
    # rescale to sfactor
    ims4 = [im.resize((int(sw/sf),int(sl/sf)), Image.NEAREST) for im in ims3]
    # return to numpy formta
    ims5 = [np.array(im.getdata()).reshape(im.size[0], im.size[1]) for im in ims4]
    #np.array(list(ims4[0].getdata())).reshape([sf*sw,sf*sl])

    #plt.imshow(ims5[0])
    #plt.show()
    ims6 = [im.reshape(int(sl*sw/sf/sf)) for im in ims5]
    ims7 = np.asarray(ims6)
    ipo = torch.tensor(ims7).float()

    outputs = net(Variable(ipo))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0) 
    correct += (predicted == labels).sum()

print("Accuracy: {}%".format(100*correct.numpy()/total))

		#while we're here, dump examples
images_o, labels_o = [], []
for i, data in enumerate(testloader):
  ip, il = data
  for p,l in zip(ip,il):
    images_o.append(p.numpy())
    labels_o.append(l)

  if i==19:
    break

fo = ddir + "/mnist_examples.pkl"
with open(fo, "wb") as f: 
    import _pickle as cPickle 
    data_dict = { "images":images_o, "labels": labels_o}
    cPickle.dump(data_dict, f)

print("Dumping training weights to disk")
weights_dict = {} 
#import ipdb; ipdb.set_trace()
for param in list(net.named_parameters()):
    print("Serializing Param", param[0])
    weights_dict[param[0]] = param[1] 
# with open("weights.pkl","wb") as f:
#     import pickle 
#     pickle.dump(weights_dict, f)
fo = ddir + "/Config-2-weights.pkl"
with open(fo, "wb") as f: 
    import _pickle as cPickle 
    cPickle.dump(weights_dict, f)
print("Finished dumping to {}".format(fo))
