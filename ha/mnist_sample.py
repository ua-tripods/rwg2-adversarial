		#### mnist_attack ####
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
		# setup directory structure
ddir = "/home/bwbell/Dropbox/UOFA/0-research/network/rwg2-adversarial/ha/"
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
weight_opt = optim.SGD(params=[net.img], lr=0.01)  

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

#fresh network for classification
net_s = Net()
weights_dict = {}
assert os.path.isfile(wfile), "Error: Invalid {} ".format(wfile)
with open(wfile, "rb") as f:
    weights_dict = pickle.load(f)
for param in net_s.named_parameters():
    if param[0] in weights_dict.keys():
        print("Copying: ", param[0])
        param[1].data = weights_dict[param[0]].data 
print("Weights Loaded!")

# initialize

# Attack each example 
count, total  = 0, 0;
i_sam = np.random.randint(0,images.__len__())

ii_lab, ii_sam, ii_cou = np.unique(labels, return_counts=True, return_index=True)
for i_sam in ii_sam:
  print(i_sam)
  iin = images[i_sam]
  cin = labels[i_sam]
  
  #for iin, cin in tqdm(zip(images, labels)):
  		# pick a random target
                  # TODO : compute Cartesian Product
  #    ctar   = random.choice( list(set([0,1,2,3,4,5,6,7,8,9]) - set([cin])) )
  
  for i in list(set([0,1,2,3,4,5,6,7,8,9]) - set([cin])):
    ctar = i
    xs, y_trues, y_preds, y_preds_adversarial, noises = [], [], [], [], []
    ox, ctrue_l, cpred_l, cpred_a_l, noise_l = [], [], [], [], []

    iin_t  = Variable(torch.FloatTensor(iin))
    cpred =  np.argmax(net_s(iin_t).data.numpy())
    bad_class = False
    print ("True class: {} | Prediction: {} | Target: {}".format(cin, cpred, ctar))
    if cin != cpred:
      bad_class = True
      print("WARNING: Bad Classification")
  
      #for i in list(set(range(0,10)) - set([cin])):
    # i = 0
    # if cin == i:
    #   i = 1
    if True:
      #ctar = i
      total = total+1
      ctar_t = Variable(torch.LongTensor([ctar]))

      # Reset image
      net.img.data = torch.zeros(1,784) 

      # If it's not correctly classified by the network, skip it
      # Optimization Loop 
      for iteration in range(1000):
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
            break 

        if iteration == 999:
          print("Warning: Hit 1000 iterations, SAVE THIS FOR REFERENCE")

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

    # uniform sampling around noise
    
    # we'll want a new network without adjustments
    cpred_ao = np.argmax(net_s(iin_t).data.numpy())
    
    iin_o = iin
    # plt.figure(0)
    # plt.imshow(iin_o.reshape(28,28))
    
    iin_a = iin + noise
    # plt.figure(1)
    # plt.imshow(iin_a.reshape(28,28))
    
    # generate a gaussian of magnitudes with mean 0 and variance |noise|
    n_sz = iin.__len__()
    mean =    np.zeros(n_sz)
    m_var = np.var(noise)
    
    
    
    
    n_iter = 500
    n_real = 5000
    v_scal = 40
    results_a = np.zeros([10,n_iter])
    results_o = np.zeros([10,n_iter])
    results_c = np.zeros([10,n_iter])
    results_z = np.zeros([10,n_iter])

    a_sigs = np.linspace(0,m_var*40,n_iter)
    count = 0
    for a_sig in a_sigs:
      cov  = a_sig*np.identity(n_sz)
      samp = np.random.multivariate_normal(mean, cov, n_real)
      m_samp = np.sqrt(np.var(samp,axis=1)/np.var(iin_a,axis=1))
      #figure(4)
      #plt.hist(m_samp)
      #plt.title("noise magnitude of the randomly added samples")
      # 5k samples
      i_samp_a = samp+iin_a[0]
      i_tens = Variable(torch.FloatTensor(i_samp_a))
      i_samp_n = samp+iin_o
      i_tens_n = Variable(torch.FloatTensor(i_samp_n))
      i_samp_c = samp
      i_tens_c = Variable(torch.FloatTensor(i_samp_c))
      i_samp_z = np.zeros(n_sz)
      i_tens_z = Variable(torch.FloatTensor(i_samp_z))
    
      cpred_ao = np.argmax(net_s(i_tens).data.numpy(),axis=1)
      a, b = np.unique(cpred_ao, return_counts=True)
      results_a[a,count] = b
      cpred_ao_n = np.argmax(net_s(i_tens_n).data.numpy(),axis=1)  
      a, b = np.unique(cpred_ao_n, return_counts=True)
      results_o[a,count] = b
      cpred_ao_c = np.argmax(net_s(i_tens_c).data.numpy(),axis=1)  
      a2, b2 = np.unique(cpred_ao_c, return_counts=True)
      results_c[a2,count] = b2
      cpred_ao_z = np.argmax(net_s(i_tens_z).data.numpy(),axis=1)  
      a3, b3 = np.unique(cpred_ao_z, return_counts=True)
      results_z[a3,count] = b3
    
      count += 1
      # illustrate the various noise configurations being added
      print("generating replication {}/{} with sigma {}".format(count,
                                                       n_iter, a_sig))   
    i_disp = np.zeros([2*28,4*28])
    i_disp[0*28:1*28,0*28:1*28] = iin_o.reshape(28,28)
    i_disp[0*28:1*28,1*28:2*28] = i_samp_n[0].reshape(28,28)
    i_disp[0*28:1*28,2*28:3*28] = iin_a[0].reshape(28,28)
    i_disp[0*28:1*28,3*28:4*28] = i_samp_a[0].reshape(28,28)
    i_disp[1*28:2*28,0*28:1*28] = i_samp_z.reshape(28,28)   
    i_disp[1*28:2*28,1*28:2*28] = i_samp_c[0].reshape(28,28)
    i_disp[1*28:2*28,2*28:3*28] = noise.reshape(28,28)
    i_disp[1*28:2*28,3*28:4*28] = (i_samp_c[0]+noise).reshape(28,28)
  
    fig= plt.figure()
    plt.imshow(i_disp)
    fig.set_size_inches(24,12)
    fo = ddir+"O{}A{}_varx{}_examp.png".format(cin, cpred_a,v_scal)
    fig.savefig(fo, dpi=100)  
    #plt.imshow(i_samp_c[0].reshape(28,28))
    #plt.show()
    
    plt.figure(num=1, figsize=(24,18), dpi=100)
    
    fig, ax= plt.subplots(3,sharex=True, sharey=True)
    fig.set_size_inches(24,12)
    
    ax[0].plot(a_sigs,results_o.T, linewidth=2.0)
    ax[0].set_ylabel("original")
    ax[0].legend([0,1,2,3,4,5,6,7,8,9],loc='upper right')
    
    ax[1].plot(a_sigs,results_a.T, linewidth=2.0)
    ax[1].set_ylabel("adversarial")
    ax[1].legend([0,1,2,3,4,5,6,7,8,9],loc='upper right')
    
    
    ax[2].plot(a_sigs,results_c.T, linewidth=2.0)
    ax[2].set_ylabel("noise")
    ax[2].legend([0,1,2,3,4,5,6,7,8,9],loc='upper right')
    # ax.subplot(2,2,4)
    # ax.plot(a_sigs,results_z.T, linewidth=2.0)
    # ax.title("zeros")
    # ax.legend([0,1,2,3,4,5,6,7,8,9],loc='upper right')
    fo = ddir+"O{}A{}_varx{}.png".format(cin, cpred_a, v_scal)
    print("generating: {}".format(fo))
    fig.subplots_adjust(hspace=0)
    fig.set_size_inches(24,18)
    fig.savefig(fo, dpi=100)

# # plt.figure(2)
# # plt.hist(cpred_ao)

# # i_s = np.random.randint(0,i_samp_a.__len__())
# # iin_s = i_samp_a[i_s]
# # # measure the magnitude of the gaussian noise that we added 
# # #n_mag =
# # # measure the noise of the original adversarial noise

# # plt.figure(3)
# # plt.imshow(iin_s.reshape(28,28))
# # plt.title("Orig:{}, Adv:{}, Final:{}".format(cin, cpred_a, cpred_ao[i_s]))
# # plt.show()

# # plt.plot(x, y, 'x')
# # plt.axis('equal')
# # plt.show()
# # for each magnitude, generate a uniformly random vector with that magnitude
# cpred_ao = np.argmax(net_s(iin_t).data.numpy())


# print("Found: {}/{} Adversarial Examples".format(count, total))
# with open("mnist_attack.pkl","wb") as f: 
#     save_dict = {"ox":ox, 
#                  "ctrue":ctrue_l,
#                  "cpred":cpred_l,
#                  "cpred_a":cpred_a_l,
#                  "noises": noise_l }
#     pickle.dump(save_dict, f) 
# print("Dumped to mnist_attack.pkl")
 
