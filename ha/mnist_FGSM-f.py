import matplotlib.pyplot as plt
import numpy as np 
import pickle 
import argparse

from   PIL import Image
import random 
import sys, os, pickle
import torch
from   torch.autograd import Variable
import torch.optim as optim 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
from   torchvision import transforms
from   tqdm import *

# adversarial robustness toolkit (ART)
# import keras.backend as k
# from keras.models import Sequential
# from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
# import numpy as np

# from art.attacks.fast_gradient import FastGradientMethod
# from art.attacks.iterative_method import BasicIterativeMethod
# from art.attacks.carlini import CarliniL2Method

# from art.classifiers import KerasClassifier
# from art.utils import load_dataset

# parser = argparse.ArgumentParser(description='Grab some inputs')
# parser.add_argument("-factor", "-fileheader", required=False)
# parser.parse_args()

# system arguments and size variables
sf = 1
if (len(sys.argv) > 1):
  sf = np.float(sys.argv[1])
fileheader = "f1"
if (len(sys.argv) > 2):
  fileheader = str(sys.argv[2])
  print(fileheader)

sl = 28
sw = 28

		#### mnist_attack ####

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

fo    = ddir + "/Config-2-weights.pkl"
wfile = ddir + "/Config-2-weights-"+fileheader+".pkl"
efile = ddir + "/mnist_examples-"+fileheader+".pkl"

# Define Attack Network
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
# class Net(nn.Module): 
#     def __init__(self, conf):
#         super(Net, self).__init__()
# 		# define the layers of the attack network
#                 #        All layers must match the training network
#                 #        except the img layer 
#         self.img = nn.Parameter(data=torch.zeros(1,conf[0]), requires_grad=True)
#         pairs = zip(conf[:-1], conf[1:])
#         self.fcs = nn.ModuleList(nn.Linear(this_layer, next_layer)
#                        for (this_layer, next_layer) in pairs)
#         # self.fcs = list(nn.Linear(this_layer, next_layer)
#         #                for (this_layer, next_layer) in pairs)
# 					 # desired number of classes.  
#     def forward(self, x):
# 		# define the activation filters that connect layers
#         x = x + self.img       # make it easy to turn this into an image
#         x = torch.clamp(x,0,1) 
#         for layer in self.fcs[:-1]:
#           x = F.relu(layer(x))
#         x = self.fcs[-1](x)
#         return x
    
		# initialize our adversarial network
# todo make this a user-parameter

conf = [int(sl*sw/sf/sf), int(100/sf/sf), 10]
nnet = Net(conf)
		# Error function for finding adversarial gradients 
output_fil = nn.CrossEntropyLoss()
#output_fil = nn.MSELoss()
		# Optimizer we'll use for gradient descent.
                #        params : the list of all parameters the
                #                 optimizer can change
                #             Default: All
                #             FGSM   : just the image
                # TODO: try plain gradient descent
#weight_opt = optim.SGD(params=[nnet.img], lr=0.001)
#weight_opt = optim.SGD(params=[nnet.img], lr=0.001)
                  #, momentum=0.5, param weight_decay=0.1)  

		# load the weights from training
wweights_dict = {}
assert os.path.isfile(wfile), "Error: Invalid {} ".format(wfile)
with open(wfile, "rb") as f:
    wweights_dict = pickle.load(f)
for param in nnet.named_parameters():
    if param[0] in wweights_dict.keys():
        print("Copying: ", param[0])
        param[1].data = wweights_dict[param[0]].data 
print("Weights Loaded!")

# Load examples
with open(efile,"rb") as f:  
    examples = pickle.load(f) 

ims1 = examples["images"]
# Reshape
ims2 = [im.reshape(sl,sw) for im in ims1]
# convert to Image
ims3 = [Image.fromarray(im) for im in ims2]
# rescale to sfactor

ims4 = [im.resize((int(sw/sf),int(sl/sf)), Image.NEAREST) for im in ims3]
# return to numpy formta
ims5 = [np.array(im.getdata()).reshape(im.size[0], im.size[1]) for im in ims4]
#np.array(list(ims4[0].getdata())).reshape([sf*sw,sf*sl])
#plt.imshow(ims5)
#plt.show()
ims6 = [im.reshape(int(sl*sw/sf/sf)) for im in ims5]
images = ims6

labels = examples["labels"]

# initialize
xs, y_trues, y_preds, y_preds_adversarial, noises = [], [], [], [], []
ox, ctrue_l, cpred_l, cstep_l, ctar_l, cpred_a_l, noise_l = {}, {}, {}, {}, {}, {}, {}

# Attack each example 
count, total  = 0, 0;
icount = 0
imcount = 0
itotal = images.__len__()*9
niter = 3000

# i_sam = np.random.randint(0,images.__len__())
# i_sam = 0
# iin = images[i_sam]
# cin = labels[i_sam]
# #oneshot:
# iin_t  = torch.tensor(iin, requires_grad=True)
# iin_o  = nnet(iin_t)
# cpred = torch.max(iin_o.data,1)[1][0]
# #cpred =  np.argmax(nnet(iin_t).data.numpy())
# bad_class = False
# print ("True class: {} | Prediction: {}".format(cin, cpred))
# if cin != cpred:
#     bad_class = True
#     print("WARNING: Bad Classification")

# l_tar = list(set(range(0,10)) - set([cin]));
# ctar = l_tar[1]
# ctar_t = Variable(torch.FloatTensor([ctar]))      

# nnet.img.data = torch.zeros(1,784) 
# # for i in list(set(range(0,10)) - set([cin])):
# #   icount += 1
# #   ctar = i
# #   total = total+1
# #   #ctar_t = Variable(torch.LongTensor([ctar]))
#   # Reset image
#       # If it's not correctly classified by the nnetwork, skip it
#       # Optimization Loop
# losses = np.zeros(niter)
# for iteration in range(niter):
#   weight_opt.zero_grad() 
#   outputs = nnet(iin_t)
#   cpred_a = np.argmax(nnet(iin_t).data.numpy())
#   #cpred_aa = torch.FloatTensor(torch.argmax(nnet(iin_t).data))
#   cpred_aa = torch.FloatTensor([cpred_a])
#   xent_loss = output_fil(cpred_aa, ctar_t) 
#   # Add regularization -- this is L2
#   adv_loss  = xent_loss + torch.mean(torch.pow(nnet.img,2))
#   # The Big Scary Important Step
#   losses[iteration] = adv_loss
#   adv_loss.backward() 
#   weight_opt.step() 
#   # keep optimizing Until classif_op == ctar_t


#   # good enough?
#   if iteration > 10:
#     deriv  = np.sum(np.abs((losses[(iteration-9):iteration] -
#              losses[(iteration-10):(iteration-1)])))/9
#     if (np.abs(deriv) < 0.01) & (np.abs(deriv) > 0.0):
#       print("Tolerance Reached")
#       break

#   if iteration == (niter - 1):
#     print("Warning: Hit {} iterations, SAVE THIS FOR REFERENCE".format(niter))

# noise = nnet.img.data.numpy()

# if ((cpred == cin) & (cpred_a == ctar)):
#   count = count+1
#   mnoises = (np.array(nnet.img.data) - iin).reshape(28,28) 
#   onoises = np.array(iin).reshape(28,28) 
#   print("{}/{}Found adv_example : Iter {}"+
#         " : Noise :{:.4f}".format(icount, itotal, iteration, 
#   np.sqrt(np.var(mnoises)/np.var(onoises))))   

#   #print("Found adv_example : Iter {}".format(iteration))
#   # store
#   ox.append(iin)
#   ctrue_l.append(cin)
#   cpred_l.append(cpred)
#   cpred_a_l.append(cpred_a)
#   noise_l.append(noise.squeeze())
# else: 
#   print("Bad Classification: Skipped")
a = np.array([label.numpy() for label in labels])
    # preliminary check for accuracy
iin_t  = torch.tensor(images, requires_grad=True).float()
iin_o  = nnet(iin_t)
cpred = torch.argmax(iin_o.data,1)
labels_a = np.array([label.numpy() for label in labels])
cpred_a = cpred.numpy()
test = np.sum(cpred_a == labels_a)
print("Accuracy: {}% : {}/{}".format(100*test/len(cpred_a), test,len(cpred_a)))


TEST = False

for iin, cin in tqdm(zip(images, labels)):

    # if (icount > 100):
    #   break
		# pick a random target
    iin_t = torch.tensor(iin, requires_grad=True).float()

    iin_o = nnet(iin_t)
    cpred = torch.argmax(iin_o.data)

    # keep what we find:
    ox[imcount] = iin
    ctrue_l[imcount]   = cin
    cpred_l[imcount]   = cpred
    cstep_l[imcount]   = 0
    ctar_l[imcount]    = []
    cpred_a_l[imcount] = []
    noise_l[imcount]   = []


    if (TEST == True) & (imcount > 10):
      break
    # find stuff that was classified correctly !

    # skip anything we don't classify correctly to begin with
    #cpred =  np.argmax(nnet(iin_t).data.numpy())
    bad_class = False
    print ("True class: {} | Prediction: {}".format(cin, cpred))
    if cin != cpred:
        bad_class = True
        print("WARNING: Bad Classification")

      
    for i in list(set(range(0,10)) - set([int(cin.numpy())])):
      icount += 1
      ctar = i
      total = total+1
      noise = Variable(torch.zeros(1,len(iin_t)),requires_grad=True)
      #ctar_t = Variable(torch.LongTensor([ctar]))
      ctar_t = Variable(torch.LongTensor([ctar]))      

      # Reset image old as of not using internal optimizer
      #nnet.img.data = torch.zeros(1,len(iin_t)) 

      # If it's not correctly classified by the nnetwork, skip it
      # Optimization Loop
      # take the sign of the gradient times epsilon and that's it. 
      losses = np.zeros(niter)

      for iteration in range(niter):
        #weight_opt.zero_grad() 
        outputs = nnet(iin_t+noise)
        cpred_a = int(torch.argmax(outputs))# nnet(iin_t).data.numpy())
        # stop once we break the classification
        # note that we can keep going here.
        # TODO play with stopping condition
        if (cpred_a == ctar):# & iteration > 30):
          break
        xent_loss = output_fil(outputs, ctar_t) 
        # Add regularization -- this is L2
        adv_loss  = xent_loss + torch.mean(torch.pow(noise,2))
        # The Big Scary Important Step
        losses[iteration] = adv_loss
        adv_loss.backward() 

        epsilon = 0.01
        grad  = torch.sign(noise.grad.data)
        noise = Variable(noise - epsilon*grad, requires_grad=True)
	# todo histogram of number of steps -- want this to be nicely distrubuted, not clustered around 1 or 3000

        # good enough?
        # if iteration > 1000:
        #   deriv  = np.sum(np.abs((losses[(iteration-9):iteration] -
        #                        losses[(iteration-10):(iteration-1)])))/9
        #   if (np.abs(deriv) < 0.0000001) & (np.abs(deriv) > 0.0):
        #     print("Tolerance Reached")
        #     break
            
        if iteration == (niter - 1):
          print("Warning: Hit {} iterations, SAVE THIS FOR REFERENCE".format(niter))

      #noise = nnet.img.data.numpy()

      if (cpred.numpy() == cin.numpy()):
        if (cpred_a != cin):
          count = count+1
          mnoises = noise.detach().numpy()# (np.array(nnet.img.data) - iin).reshape(int(sw/sf),int(sl/sf)) 
          onoises = np.array(iin)# .reshape(int(sw/sf),int(sl/sf)) 
          print("{}/{}Found adv_example : Iter {}: Noise :{:.4f}".format(icount, itotal, iteration, np.sqrt(np.var(mnoises)/np.var(onoises))))   
          if (cpred_a != ctar):
            print("Hit Wrong Target: {} to {} instead of {}".format(cin.numpy(), cpred_a, ctar))


          #print("Found adv_example : Iter {}".format(iteration))
          # store
          cstep_l[imcount] = iteration
          ctar_l[imcount].append(ctar)
          cpred_a_l[imcount].append(cpred_a)
          noise_l[imcount].append(noise.squeeze())

        else:
          ctar_l[imcount].append(ctar)
          cpred_a_l[imcount].append(cpred_a)
          noise_l[imcount].append(noise.squeeze())
          print("Failed to find Adversarial Example")
      else:
        print("Bad Classification: {} instead of {} Skipped".format(cpred, cin))

    imcount += 1
print("Found: {}/{} Adversarial Examples".format(count, total))
fo = ddir+"mnist_attack-L2Loss-"+fileheader+".pkl"
with open(fo,"wb") as f: 
    save_dict = {"ox":ox, 
                 "ctrue":ctrue_l,
                 "cpred":cpred_l,
                 "ctar":ctar_l,                 
                 "cpred_a":cpred_a_l,
                 "noises": noise_l }
    pickle.dump(save_dict, f) 
print("Dumped to: {}".format(fo))
 

# TODO Add .csv writer
