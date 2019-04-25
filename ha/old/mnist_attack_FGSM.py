import numpy as np 
import torch
from torch.autograd import Variable
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torch.optim as optim 
from torchvision import transforms
from tqdm import *
import matplotlib.pyplot as plt
import pickle 
import random 
import sys, os, pickle

# logging
import logging
FORMAT = '%(asctime)-15s'
logging.basicConfig(filename='mnist_attack.log', mode='w',level=logging.DEBUG,format=FORMAT)

logging.info('Starting Log')
logging.debug('Debug')
logging.warning('Warnings')
logging.error('Errors')
		#### mnist_attack ####

wfile = "/home/bwbell/Dropbox/UOFA/0-research/network/ha/weights.pkl"

# Define Attack Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
		# define the layers of the attack network
                #        All layers must match the training network
                #        except the img layer 
        self.img = nn.Parameter(data=torch.zeros(1,28*28), requires_grad=True)
        self.fc1 = nn.Linear(28*28, 392) # size of the image vector
        self.fc2 = nn.Linear(  392, 196) # reduce the data several times
        self.fc3 = nn.Linear(  196, 98)  # ...
        self.fc4 = nn.Linear(   98, 10)  # final output must match the
					 # desired number of classes.  

    def forward(self, x):
		# define the activation filters that connect layers
        x = x + self.img       # make it easy to turn this into an image
        x = torch.clamp(x,0,1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
		# initialize our adversarial network
net = Net()
		# Error function for finding adversarial gradients 
output_fil = nn.CrossEntropyLoss()
		# Optimizer we'll use for gradient descent.
                #        params : the list of all parameters the
                #                 optimizer can change
                #             Default: All
                #             FGSM   : just the image
                # TODO: try plain gradient descent
weight_opt = optim.SGD(params=[net.img], lr=0.001,momentum=0.9,
                       weight_decay=0.00001)  

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
with open("/home/bwbell/Dropbox/UOFA/0-research/network/ha/mnist_examples.pkl","rb") as f:  
    examples = pickle.load(f) 
images = examples["images"]
labels = examples["labels"]

# initialize
xs, y_trues, y_preds, y_preds_adversarial, noises = [], [], [], [], []
ox, ctrue_l, cpred_l, cpred_a_l, noise_l = [], [], [], [], []

# Attack each example 
n_iter = 3000;
count, total  = 0, 0;
progress = 0;
p_total = 5120*10;
for iin, cin in tqdm(zip(images, labels)):
		# pick a random target
                # TODO : compute Cartesian Product
#    ctar   = random.choice( list(set([0,1,2,3,4,5,6,7,8,9]) - set([cin])) )

    iin_t  = Variable(torch.FloatTensor(iin))
    cpred =  np.argmax(net(iin_t).data.numpy())
    bad_class = False
    print ("True class: {} | Prediction: {}".format(cin, cpred))
    if cin != cpred:
        bad_class = True
        print("WARNING: Bad Classification")

    for i in list(set(range(0,10)) - set([cin])):
      if (progress%100==0):
         logging.log("Progress: {}/{}".format(progress,p_total))

      progress = progress+1
      ctar = i
      total = total+1
      ctar_t = Variable(torch.LongTensor([ctar]))

      # Reset image
      net.img.data = torch.zeros(1,784) 

      # If it's not correctly classified by the network, skip it
      # Optimization Loop 
      for iteration in range(n_iter):
        weight_opt.zero_grad() 
        outputs = net(iin_t)
        xent_loss = output_fil(outputs, ctar_t) 
        # Add regularization -- this is L2
        adv_loss  = xent_loss + torch.mean(torch.pow(net.img,2))
        # The Big Scary Important Step
        adv_loss.backward() 
        weight_opt.step() 
        # keep optimizing Until classif_op == ctar_t
        cpred_a = np.argmax(net(iin_t).data.numpy())
        if cpred_a == ctar:
            mnoises = (np.array(net.img.data) - iin).reshape(28,28) 
            onoises = np.array(iin).reshape(28,28) 
            #print("Found adv_example : Iter {} : Noise : {:.4f}".format(
            #      iteration, np.sqrt(np.var(mnoises)/np.var(onoises))))  
            break 

        if iteration == n_iter-1:
          print("Warning: Hit {} iterations, SAVE THIS FOR REFERENCE".format(n_iter))

      noise = net.img.data.numpy()

      if cpred == cin:
        count = count+1
        # store
        ox.append(iin)
        ctrue_l.append(cin)
        cpred_l.append(cpred)
        cpred_a_l.append(cpred_a)
        noise_l.append(noise.squeeze())
      else: 
        print("Bad Classification: Skipped")

print("Found: {}/{} Adversarial Examples".format(count, total))
with open("mnist_attack.pkl","wb") as f: 
    save_dict = {"ox":ox, 
                 "ctrue":ctrue_l,
                 "cpred":cpred_l,
                 "cpred_a":cpred_a_l,
                 "noises": noise_l }
    pickle.dump(save_dict, f) 
print("Dumped to mnist_attack.pkl")
 
