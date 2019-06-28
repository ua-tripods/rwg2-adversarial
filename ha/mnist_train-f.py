import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image
import random
import sys, os, pickle
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from tqdm import *

#### mnist_train.py ####

# Sort out Directories
sfile = "C:/Users/Nexus/Google Drive/dropbox/Dropbox/UOFA/0-research/network/rwg2-adversarial/ha/imnet_fgsm.py"
lfile = "/home/bwbell/Dropbox/UOFA/0-research/network/rwg2-adversarial/ha/imnet_fgsm.py"
if (os.path.isfile(sfile)):
    ddir = "C:/Users/Nexus/Desktop/Adversarial-Examples-in-PyTorch2/mnist_scale"
    odir = "C:/Users/Nexus/Google Drive/dropbox/Dropbox/UOFA/0-research/network/imnet_examples_nexus"
    ldir = "C:/Users/Nexus/Desktop/Adversarial-Examples-in-PyTorch2/ilsvrc12_imnet_labels"
elif (os.path.isfile(lfile)):
    ddir = "/home/bwbell/Desktop/Adversarial-Examples-in-PyTorch/mnist_scale"
    mdir = "/home/bwbell/Desktop/Adversarial-Examples-in-PyTorch/mnist2"
    odir = "/home/bwbell/Dropbox/UOFA/0-research/network/imnet_examples"
    # directory for labels in ilsvrc_12
    ldir = "/home/bwbell/Desktop/Adversarial-Examples-in-PyTorch/ilsvrc12_imnet_labels"
elif (os.path.isfile(lfile)):
    ddir = "/home/bwbell/Adversarial-Examples-in-PyTorch/mnist_scale"
    mdir = "/home/bwbell/Adversarial-Examples-in-PyTorch/mnist2"
    odir = "/home/bwbell/Adversarial-Examples-in-Pytorch/imnet_examples"
else:
    ddir = "/home/bwbell/Adversarial-Examples-in-PyTorch/mnist_scale"
    os.makedirs(ddir, exist_ok=True)
    mdir = "/home/bwbell/Adversarial-Examples-in-PyTorch/mnist2"
    os.makedirs(mdir, exist_ok=True)
    odir = "/home/bwbell/Adversarial-Examples-in-PyTorch/mnist_scale"
    os.makedirs(odir, exist_ok=True)

# list of files for review
ddirs = os.listdir(ddir)
ddirs.sort()


class Net(nn.Module):
    def __init__(self, conf):
        super(Net, self).__init__()
        # define the layers of the network -- place any
        #        network you want here
        pairs = zip(conf[:-1], conf[1:])
        self.fcs = nn.ModuleList(nn.Linear(this_layer, next_layer)
                                 for (this_layer, next_layer) in pairs)

    def forward(self, x):
        # define the activation filters that connect layers
        for layer in self.fcs[:-1]:
            x = F.relu(layer(x))
        x = self.fcs[-1](x)
        return x


sf = 1
if (len(sys.argv) > 1):
  sf = np.float(sys.argv[1])
fileheader = "f1"
if (len(sys.argv) > 2):
  fileheader = str(sys.argv[2])
  print(fileheader)

sl = 28
sw = 28

conf = [int(sl * sw / sf / sf), int(100 / sf / sf), 10]
net = Net(conf)
output_fil = nn.CrossEntropyLoss()  # error function
# pick an optimizer -- pytorch has many
# TODO : Look up why momentum works
weight_opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9,
                       weight_decay=0.00001)


# dataloaders copied from github.com/akshaychawla
def flat_trans(x):
    x.resize_(28 * 28)
    return x


mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(flat_trans)])
imsize = (int(sw / sf), int(sl / sf))
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
toten = transforms.ToTensor()
traindata = torchvision.datasets.MNIST(root=mdir, train=True,
                                       download=True,
                                       transform=mnist_transform)
trainloader = torch.utils.data.DataLoader(traindata, batch_size=256,
                                          shuffle=True, num_workers=2)
testdata = torchvision.datasets.MNIST(root=mdir, train=False,
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
        # Reshape
        ims2 = images.reshape(len(ip), sl, sw)  # [im.numpy().reshape(sl,sw) for im in images]
        # convert to Image
        ims3 = [Image.fromarray(im) for im in ims2]
        # rescale to sfactor
        ims4 = [im.resize((int(sw / sf), int(sl / sf)), Image.NEAREST) for im in ims3]
        # return to numpy formta
        ims5 = [np.array(im.getdata()).reshape(im.size[0], im.size[1]) for im in ims4]
        # np.array(list(ims4[0].getdata())).reshape([sf*sw,sf*sl])

        # plt.imshow(ims5[0])
        # plt.show()
        ims6 = [im.reshape(int(sl * sw / sf / sf)) for im in ims5]
        ims7 = np.asarray(ims6)
        ipo = torch.tensor(ims7).float()

        weight_opt.zero_grad()
        out = net(ipo)
        loss = output_fil(out, il)
        loss.backward()
        weight_opt.step()
        #closs += loss.data[0] # depracated
        closs += loss.item()
        print('Epoch: {}'.format(epoch))

print("Training Complete")

# TEST 
correct = 0.0
total = 0
for data in testloader:
    ip, labels = data
    images = np.array(ip)
    # Reshape
    ims2 = images.reshape(len(ip), sl, sw)
    # convert to Image
    ims3 = [Image.fromarray(im) for im in ims2]
    # rescale to sfactor
    ims4 = [im.resize((int(sw / sf), int(sl / sf)), Image.NEAREST) for im in ims3]
    # return to numpy formta
    ims5 = [np.array(im.getdata()).reshape(im.size[0], im.size[1]) for im in ims4]
    # np.array(list(ims4[0].getdata())).reshape([sf*sw,sf*sl])

    # plt.imshow(ims5[0])
    # plt.show()
    ims6 = [im.reshape(int(sl * sw / sf / sf)) for im in ims5]
    ims7 = np.asarray(ims6)
    ipo = torch.tensor(ims7).float()

    outputs = net(Variable(ipo))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print("Accuracy: {}%".format(100 * correct.numpy() / total))

# while we're here, dump examples
images_o, labels_o = [], []
for i, data in enumerate(testloader):
    ip, il = data
    for p, l in zip(ip, il):
        images_o.append(p.numpy())
        labels_o.append(l)

    if i == 19:
        break

fo = ddir + "/mnist_examples-"+fileheader+".pkl"
with open(fo, "wb") as f:
    import _pickle as cPickle

    data_dict = {"images": images_o, "labels": labels_o}
    cPickle.dump(data_dict, f)

print("Dumping training weights to disk")
weights_dict = {}
# import ipdb; ipdb.set_trace()
for param in list(net.named_parameters()):
    print("Serializing Param", param[0])
    weights_dict[param[0]] = param[1]
# with open("weights.pkl","wb") as f:
#     import pickle 
#     pickle.dump(weights_dict, f)
fo = ddir + "/Config-2-weights-"+fileheader+".pkl"
with open(fo, "wb") as f:
    import _pickle as cPickle

    cPickle.dump(weights_dict, f)
print("Finished dumping to {}".format(fo))
