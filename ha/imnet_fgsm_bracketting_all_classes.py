import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torch.optim as optim 
from torchvision import transforms
from tqdm import *
from scipy.sparse import identity
from scipy.stats import multivariate_normal
import torchvision.models 
import pickle
import pandas as pd
import os
from timeit import default_timer as timer
from datetime import timedelta
from collections import defaultdict

# constants
imsize = (224, 224)
n_iter = 30
n_real = 10
v_scal = 40

		# directory for validation images
sfile = "C:/Users/Nexus/Google Drive/dropbox/Dropbox/UOFA/0-research/network/rwg2-adversarial/ha/imnet_fgsm.py"
if (os.path.isfile(sfile)):
  ddir = "C:/Users/Nexus/Desktop/Adversarial-Examples-in-PyTorch2/ILSVRC2012_img_val"
  odir = "C:/Users/Nexus/Google Drive/dropbox/Dropbox/UOFA/0-research/network/imnet_examples_nexus"
  ldir = "C:/Users/Nexus/Desktop/Adversarial-Examples-in-PyTorch2/ilsvrc12_imnet_labels"
else:  
  ddir = "/home/bwbell/Desktop/Adversarial-Examples-in-PyTorch/ilsvrc12_val_images"
  odir = "/home/bwbell/Dropbox/UOFA/0-research/network/imnet_examples"
		# directory for labels in ilsvrc_12
  ldir = "/home/bwbell/Desktop/Adversarial-Examples-in-PyTorch/ilsvrc12_imnet_labels"
		# list of files for review
ddirs = os.listdir(ddir)
ddirs.sort()
# test to make sure we don't have erronious stuff in the image directory
count = 0
for d in ddirs:
  if d[-5:] == ".JPEG":
    count+=1
if count!=len(ddirs):
  print("warning, too many images: {}/{}".format(count, len(ddirs)))



# functions
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
def image_loader(image_name):
    """load image, returns tensor"""
    image = Image.open(image_name)
    image = image.convert("RGB") # Auto remove the "alpha" channel from png image
    image = loader(image).float()
    image = normalize(image).float()
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image 

def n_normalize(image_data):
  c = image_data
  e = (c+np.abs(np.min(c)))
  f_x = e/np.max(np.abs(e))
  return f_x

def bracketting (image, net, n, tol, n_real):
		# start with same magnitude noise as image
  a_var0 = np.var(x.detach().numpy())/4
  a_var = a_var0
  l_var = 0
  u_var = a_var*2
  a_vars = np.zeros(n)
  # Adversarial image plus noise counts
  a_counts = np.zeros(n)
  count = 0
  rap = np.zeros([n_real,n_iter])
		# grab the classification of the image under the network
  y_a = np.argmax(net.forward(Variable(image)).data.numpy())
  print("Original Classification:\t({}) {}".format(
         y_a,
         ImageNet_mapping[str(y_a)]))
  # check that u_var is high enough
  print("Is u_var is high enough?")
  
  samp = np.random.normal(0,np.sqrt(u_var),m.shape)
  samp_t = Variable(torch.Tensor(samp))
  pert_a = image.data + samp_t
  start = timer()
  image_a = net.forward(Variable(pert_a)).data.numpy()
  end = timer()
  image_as = np.argmax(image_a,axis=1)
  print("u_var:{}, count:{}".format(u_var, np.sum(image_as == y_a)))
  while(np.sum(image_as == y_a) > n_real*tol/2):
    u_var = u_var*2
    samp = np.random.normal(0,np.sqrt(u_var),m.shape)
    samp_t = Variable(torch.Tensor(samp))
    pert_a = image.data + samp_t
    start = timer()
    image_a = net.forward(Variable(pert_a)).data.numpy()
    end = timer()
    image_as = np.argmax(image_a,axis=1)
    print("u_var:{}, count:{}".format(u_var, np.sum(image_as == y_a)))
    
 
  # perform the bracketing 
  for i in range(0,n): #, a_var in enumerate(a_vars):
    count+=1
    print("Starting iteration: {}, sample variance: {}".format(count, a_var))
  		# compute sample and its torch tensor
    samp = np.random.normal(0,np.sqrt(a_var),m.shape)
    samp_t = Variable(torch.Tensor(samp))
    pert_a = image.data + samp_t
    start = timer()
    image_a = net.forward(Variable(pert_a)).data.numpy()
    end = timer()
    print("Time For {}: {}".format(n_real,timedelta(seconds=end-start)))
    image_as = np.argmax(image_a,axis=1)
    rap[:,i] = image_as

    a_counts[i] = np.sum(image_as == y_a)
    a_vars[i] = a_var # save old variance

    print("count:{}, interval: [{},{}]".format(a_counts[i], l_var, u_var))
		#floor and ceiling surround number
    if ((a_counts[i] <= np.ceil(n_real*tol)) & (a_counts[i] > np.floor(n_real*tol))):
        return {"a_vars":a_vars, "a_counts":a_counts, "a_var":a_var}        
    elif (a_counts[i] < n_real*tol): # we're too high
        print("We're too high, going from {} to {}".format(a_var, (a_var+l_var)/2)) 
        u_var = a_var
        a_var = (a_var + l_var)/2
    elif (a_counts[i] >= n_real*tol): # we're too low
        l_var = a_var
        print("We're too low,  going from {} to {}".format(a_var,(u_var+a_var)/2))
        a_var = (u_var + a_var)/2
        #u_var = u_var*2

  return {"a_vars":a_vars, "a_counts":a_counts, "a_var":a_var}

# import labels for validation images
fi = ldir + "/val.txt"
il_lab = pd.read_csv(fi,delimiter=" ", header=None)
il_lab.columns = ["img","lab"]
il_dict = il_lab.set_index("img").T.to_dict('list')

# import label names
fi = ldir + "/labels.json"
with open(fi,"r") as f: 
    import json 
    ImageNet_mapping = json.loads(f.read())

# import Pretrained VGG16 model
vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16.eval() # disable dropout, batchnorm
SoftmaxWithXent = nn.CrossEntropyLoss()
print (".. loaded pre-trained vgg16")

#setup samples
xs, y_trues, y_preds, noises, y_preds_adversarial = [], [], [], [], []

# Grab a representative image
# i = 2899
# i = 11111
# i = 2899
# i_name = ddirs[i]
# c_cor = ImageNet_mapping[str(il_dict[i_name][0])]
# print("Checking image {} \n\tcorrect class: {}".format(i_name, c_cor))
# x = Variable(image_loader(ddir+"/"+i_name),requires_grad=True)

# # plot preparation:
# fig= plt.figure()
# c = n_normalize(x.detach().numpy()[0])
# plt.imshow(np.swapaxes(c,0,2))
# fig.set_size_inches(24,12)
# fo = odir+"/"+i_name[:-5]+"-rrp.png"
# fig.savefig(fo, dpi=100)  




#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# grab (some) sample(s) of the adversarial class

for a_class in range(0,1000):
  a_imgs = li_dict[a_class]
  b3 = {}; b3v = {}
  # bracket all the images of the adversarial class
  
  count = 0
  for a_img in a_imgs:
    count+=1
    print("progress:{}/{}".format(count, len(a_imgs)))
    xs = Variable(image_loader(ddir+"/"+a_img),requires_grad=True)
    b3[a_img] = bracketting(xs,vgg16, n_iter, .68, n_real)
    b3v[a_img] = b3[a_img]["a_var"]
  
  b3v = list()
  for item in b3.items():
    b3v.append(item[1]["a_var"])
    print(item[1]["a_var"])
  
  b3vv = np.array(b3v)
    
  
  fo = odir+"/"+a_class+"-b3.pickle"
  with open(fo, 'wb') as handle:
      pickle.dump(b3, handle, protocol=pickle.HIGHEST_PROTOCOL)
  with open(fo, 'rb') as handle:
      b_test = pickle.load(handle)
      
  b3v_test = list()
  for item in b_test.items():
    b3v_test.append(item[1]["a_var"])
    #print(item[1]["a_var"])
  # b3v_test
  fo = odir+"/"+a_class+"-vars.pickle"
  with open(fo, 'wb') as handle:
      pickle.dump(b3, handle, protocol=pickle.HIGHEST_PROTOCOL)
  with open(fo, 'rb') as handle:
      b_test = pickle.load(handle)
  
    
# bracket the regular image
b1 = bracketting(x,vgg16, n_iter, .68, n_real)
# bracket the advversarial image
b2 = bracketting(adv_x,vgg16, n_iter, .68, n_real)

b3 = {}; b3v = {}
# bracket all the images of the adversarial class

count = 0
for a_img in a_imgs:
  count+=1
  print("progress:{}/{}".format(count, len(a_imgs)))
  xs = Variable(image_loader(ddir+"/"+a_img),requires_grad=True)
  b3[a_img] = bracketting(xs,vgg16, n_iter, .68, n_real)
  b3v[a_img] = b3[a_img]["a_var"]

b3v = list()
for item in b3.items():
  b3v.append(item[1]["a_var"])
  print(item[1]["a_var"])

b3vv = np.array(b3v)
  

fo = odir+"/"+i_name[:-5]+"-b3.pickle"
with open(fo, 'wb') as handle:
    pickle.dump(b3, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(fo, 'rb') as handle:
    b_test = pickle.load(handle)
    
b3v_test = list()
for item in b_test.items():
  b3v_test.append(item[1]["a_var"])
  #print(item[1]["a_var"])
# b3v_test
fo = odir+"/"+a_class+"-vars.pickle"
with open(fo, 'wb') as handle:
    pickle.dump(b3, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(fo, 'rb') as handle:
    b_test = pickle.load(handle)


b3vv = np.array(b3v)

