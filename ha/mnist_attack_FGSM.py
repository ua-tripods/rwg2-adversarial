import numpy as np 
import torch
from torch.autograd import Variable
import torch.nn as nn 
import torch.nn.functional as F 
#import torchvision 
import torch.optim as optim 
#from torchvision import transforms
from tqdm import *
import matplotlib.pyplot as plt
import pickle 
import random 
import sys, os, pickle
		#### mnist_attack ####

ddir = "/home/bwbell/"
#ddir = "C:/Users/DuxLiteratum/Google Drive/dropbox/"
#ddir = "C:/Users/Nexus/Google Drive/dropbox/"
wfile = ddir+"Dropbox/UOFA/0-research/network/rwg2-adversarial/ha/Config-2-weights.pkl"
efile = ddir+"Dropbox/UOFA/0-research/network/rwg2-adversarial/ha/mnist_examples.pkl"
#wfile = "/home/bwbell/Dropbox/UOFA/0-research/network/ha/weights.pkl"

# Define Attack Network
class Net(nn.Module): 
    def __init__(self, conf):
        super(Net, self).__init__()
		# define the layers of the attack network
                #        All layers must match the training network
                #        except the img layer 
        self.img = nn.Parameter(data=torch.zeros(1,conf[0]), requires_grad=True)
        pairs = zip(conf[:-1], conf[1:])
        self.fcs = list(nn.Linear(this_layer, next_layer)
                       for (this_layer, next_layer) in pairs)
					 # desired number of classes.  

    def forward(self, x):
		# define the activation filters that connect layers
        x = x + self.img       # make it easy to turn this into an image
        x = torch.clamp(x,0,1) 
        for layer in self.fcs[:-1]:
          x = F.relu(layer(x))
        x = self.fcs[-1](x)
        return x
    
		# initialize our adversarial network
conf = [28*28, 100, 25, 10]
net = Net(conf)
		# Error function for finding adversarial gradients 
#output_fil = nn.CrossEntropyLoss()
output_fil = nn.MSELoss()
		# Optimizer we'll use for gradient descent.
                #        params : the list of all parameters the
                #                 optimizer can change
                #             Default: All
                #             FGSM   : just the image
                # TODO: try plain gradient descent
weight_opt = optim.SGD(params=[net.img], lr=0.001)
weight_opt = optim.SGD(params=[net.img], lr=0.001)
                  #, momentum=0.5, param weight_decay=0.1)  

		# load the weights from training
weights_dict = {}
assert os.path.isfile(wfile), "Error: Invalid {} ".format(wfile)
with open(wfile, "rb") as f:
    weights_dict = pickle.load(f)
for param in net.named_parameters():
    if param[0] in weights_dict.keys():
        print("Copying: ", param[0])
        param[1].data = weights_dict[param[0]].data 
print("Weights Loaded!")

# Load examples
with open(efile,"rb") as f:  
    examples = pickle.load(f) 
images = examples["images"]
labels = examples["labels"]

# initialize
xs, y_trues, y_preds, y_preds_adversarial, noises = [], [], [], [], []
ox, ctrue_l, cpred_l, cpred_a_l, noise_l = [], [], [], [], []

# Attack each example 
count, total  = 0, 0;
icount = 0
itotal = images.__len__()*10
niter = 3000

# i_sam = np.random.randint(0,images.__len__())
# i_sam = 0
# iin = images[i_sam]
# cin = labels[i_sam]
# #oneshot:
# iin_t  = torch.tensor(iin, requires_grad=True)
# iin_o  = net(iin_t)
# cpred = torch.max(iin_o.data,1)[1][0]
# #cpred =  np.argmax(net(iin_t).data.numpy())
# bad_class = False
# print ("True class: {} | Prediction: {}".format(cin, cpred))
# if cin != cpred:
#     bad_class = True
#     print("WARNING: Bad Classification")

# l_tar = list(set(range(0,10)) - set([cin]));
# ctar = l_tar[1]
# ctar_t = Variable(torch.FloatTensor([ctar]))      

# net.img.data = torch.zeros(1,784) 
# # for i in list(set(range(0,10)) - set([cin])):
# #   icount += 1
# #   ctar = i
# #   total = total+1
# #   #ctar_t = Variable(torch.LongTensor([ctar]))
#   # Reset image
#       # If it's not correctly classified by the network, skip it
#       # Optimization Loop
# losses = np.zeros(niter)
# for iteration in range(niter):
#   weight_opt.zero_grad() 
#   outputs = net(iin_t)
#   cpred_a = np.argmax(net(iin_t).data.numpy())
#   #cpred_aa = torch.FloatTensor(torch.argmax(net(iin_t).data))
#   cpred_aa = torch.FloatTensor([cpred_a])
#   xent_loss = output_fil(cpred_aa, ctar_t) 
#   # Add regularization -- this is L2
#   adv_loss  = xent_loss + torch.mean(torch.pow(net.img,2))
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

# noise = net.img.data.numpy()

# if ((cpred == cin) & (cpred_a == ctar)):
#   count = count+1
#   mnoises = (np.array(net.img.data) - iin).reshape(28,28) 
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


for iin, cin in tqdm(zip(images, labels)):
    if (icount > 100):
      break
		# pick a random target
                # TODO : compute Cartesian Product
#    ctar   = random.choice( list(set([0,1,2,3,4,5,6,7,8,9]) - set([cin])) )

    iin_t  = torch.tensor(iin, requires_grad=True)
    iin_o  = net(iin_t)
    cpred = torch.max(iin_o.data,1)[1][0]
    #cpred =  np.argmax(net(iin_t).data.numpy())
    bad_class = False
    print ("True class: {} | Prediction: {}".format(cin, cpred))
    if cin != cpred:
        bad_class = True
        print("WARNING: Bad Classification")

    for i in list(set(range(0,10)) - set([cin])):
      icount += 1
      ctar = i
      total = total+1
      #ctar_t = Variable(torch.LongTensor([ctar]))
      ctar_t = Variable(torch.FloatTensor([ctar]))      

      # Reset image
      net.img.data = torch.zeros(1,784) 

      # If it's not correctly classified by the network, skip it
      # Optimization Loop
      losses = np.zeros(niter)
      for iteration in range(niter):
        weight_opt.zero_grad() 
        outputs = net(iin_t)
        cpred_a = np.argmax(net(iin_t).data.numpy())
        #cpred_aa = torch.FloatTensor(torch.argmax(net(iin_t).data))
        cpred_aa = torch.FloatTensor([cpred_a])
        xent_loss = output_fil(cpred_aa, ctar_t) 
        # Add regularization -- this is L2
        adv_loss  = xent_loss + torch.mean(torch.pow(net.img,2))
        # The Big Scary Important Step
        losses[iteration] = adv_loss
        adv_loss.backward() 
        weight_opt.step() 
        # keep optimizing Until classif_op == ctar_t


        # good enough?
        if iteration > 10:
          deriv  = np.sum(np.abs((losses[(iteration-9):iteration] -
                               losses[(iteration-10):(iteration-1)])))/9
          if (np.abs(deriv) < 0.01) & (np.abs(deriv) > 0.0):
            print("Tolerance Reached")
            break
            
        if iteration == (niter - 1):
          print("Warning: Hit {} iterations, SAVE THIS FOR REFERENCE".format(niter))

      noise = net.img.data.numpy()

      if ((cpred == cin) & (cpred_a == ctar)):
        count = count+1
        mnoises = (np.array(net.img.data) - iin).reshape(28,28) 
        onoises = np.array(iin).reshape(28,28) 
        print("{}/{}Found adv_example : Iter {}"+
              " : Noise :{:.4f}".format(icount, itotal, iteration, 
        np.sqrt(np.var(mnoises)/np.var(onoises))))   

        #print("Found adv_example : Iter {}".format(iteration))
        # store
        ox.append(iin)
        ctrue_l.append(cin)
        cpred_l.append(cpred)
        cpred_a_l.append(cpred_a)
        noise_l.append(noise.squeeze())
      else: 
        print("Bad Classification: Skipped")

print("Found: {}/{} Adversarial Examples".format(count, total))
fo = "mnist_attack-L2Loss.pkl"
with open(fo,"wb") as f: 
    save_dict = {"ox":ox, 
                 "ctrue":ctrue_l,
                 "cpred":cpred_l,
                 "cpred_a":cpred_a_l,
                 "noises": noise_l }
    pickle.dump(save_dict, f) 
print("Dumped to mnist_attack.pkl")
 

# TODO Add .csv writer
