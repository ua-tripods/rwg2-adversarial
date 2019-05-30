# imnet_fgsm.py
# included packages
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

		# directory for validation images
# TODO Check OS
sfile = "C:/Users/Nexus/Google Drive/dropbox/Dropbox/UOFA/0-research/network/rwg2-adversarial/ha/imnet_fgsm_example.py"
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


# constants
imsize = (224, 224)
n_iter = 100
n_real = 4
v_scal = 40


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



##### Generation Starting
# Grab a (random) representative image
# Need
#     - image
for indicew in range(0,n_iter):
  if (indicew == 0):
    i = 2899
  else: 
    i = np.random.choice(np.arange(0,50000), 1)[0]

  i_name = ddirs[i]
  c_cor = ImageNet_mapping[str(il_dict[i_name][0])]
  print("Iteration: {}/{}".format(indicew, n_iter))
  print("Checking image {} \n\tcorrect class: {}".format(i_name, c_cor))
  x = Variable(image_loader(ddir+"/"+i_name),requires_grad=True)
  
  #     - original class
  output = vgg16.forward(x)
  y_p = Variable(torch.LongTensor(np.array([output.data.numpy().argmax()])), requires_grad = False)
  		# true class
  y_t = il_dict[i_name]
  
  print("Predicted:\t({}) {}\nTrue:\t\t({}) {}".format(
    y_p.detach().numpy()[0],
    ImageNet_mapping[str(y_p.detach().numpy()[0])],
    y_t[0],
    ImageNet_mapping[str(y_t[0])]))
  
  #     - adversarial noise
  		# compute semblance of adversarial attack
  loss = SoftmaxWithXent(output, y_p)
  loss.backward()
  epsilon = 0.02
  x_grad     = torch.sign(x.grad.data)
  
  adv_n = epsilon*x_grad
  	# adv_variance 
  np.var(adv_n.numpy())
  
  #     - regular noise
  a_var = 0.3 
  m = np.zeros([1,3,224,224])
  samp = np.random.normal(0,np.sqrt(a_var),m.shape)
  samp_t = Variable(torch.Tensor(samp))
  # samp2 = np.random.normal(0,np.sqrt(a_var),m2.shape)
  # samp2_t = Variable(torch.Tensor(samp2))
  
  # v_samp = (samp - np.min(samp))/(np.max(samp)-np.min(samp))
  # v_norm = np.linalg.norm(v_samp[0])
  # plt.hist(samp.flatten(),100)
  # plt.imshow(v_samp[0].swapaxes(0,2))
  # plt.show()
  
  
  #     - adversarial image
  adv_x = x.data + adv_n  # we do not know the min/max because of torch's own stuff
  
  
  #     - image plus noise
  pert_s = x.data + samp_t
  pert_a = adv_x.data + samp_t
  adv_sv = pert_s.detach().numpy()[0]
  noutput = vgg16.forward(pert_s)
  y_on = Variable(torch.LongTensor(np.array([noutput.data.numpy().argmax()])), requires_grad = False)
  anoutput = vgg16.forward(pert_a)
  y_an = Variable(torch.LongTensor(np.array([anoutput.data.numpy().argmax()])), requires_grad = False)
  
  
  
  #     - adversarial class
  y_a = np.argmax(vgg16.forward(Variable(adv_x)).data.numpy())
  print("Adversarial:\t({}) {}".format(
    y_a,
    ImageNet_mapping[str(y_a)]))
  
  #     - reg+noise class
  # y_d = vgg16.forward(Variable(pert_s)).data.numpy()
  # y_a = np.argmax(y_d,axis=1)
  # #rrp[:,i] = y_a
  # #     - adv+noise class
  # y_d = vgg16.forward(Variable(pert_a)).data.numpy()
  # y_a = np.argmax(y_d,axis=1)
  
  
  # plot
  #original- original+noise - adversarial - adversarial+noise
  io  = n_normalize(x.detach().numpy()[0]).swapaxes(0,2).swapaxes(0,1)
  ion = n_normalize(pert_s.detach().numpy()[0]).swapaxes(0,2).swapaxes(0,1)
  ia  = n_normalize(adv_x.detach().numpy()[0]).swapaxes(0,2).swapaxes(0,1)
  ian = n_normalize(pert_a.detach().numpy()[0]).swapaxes(0,2).swapaxes(0,1)
  vio = np.var(io)
  
  no  = m[0].swapaxes(0,2).swapaxes(0,1)
  non = n_normalize(samp_t.detach().numpy()[0]).swapaxes(0,2).swapaxes(0,1)
  		# blown up by a factor of 10
  na  = epsilon*n_normalize(adv_n.detach().numpy()[0]).swapaxes(0,2).swapaxes(0,1)
  nan = non+na
  
  vno  = np.var(no)
  vnon = np.var(non)
  vna  = np.var(na)
  vnan = np.var(nan)
  
  #blank - noise - adversarial - adversarial + noise
  
  
  fig= plt.figure()
  fig, ax= plt.subplots(2,4,sharex=True, sharey=True)
  fig.set_size_inches(24,12)
  
  
  im = ax[0,0].imshow(io)
  ax[0,0].set_title("original: ({}){}\ntrue:({}){}".format(
    y_p.detach().numpy()[0],
    ImageNet_mapping[str(y_p.detach().numpy()[0])][:20],
    y_t[0],
    ImageNet_mapping[str(y_t[0])][:20]))
  
  im = ax[0,1].imshow(ion)
  ax[0,1].set_title("+noise: ({}){}\ntrue:({}){}".format(
    y_on.detach().numpy()[0],
    ImageNet_mapping[str(y_on.detach().numpy()[0])][:20],
    y_t[0],
    ImageNet_mapping[str(y_t[0])][:20]))
  im = ax[0,2].imshow(ia)
  ax[0,2].set_title("adversarial: ({}){}\ntrue:({}){}".format(
    y_a,
    ImageNet_mapping[str(y_a)][:20],
    y_t[0],
    ImageNet_mapping[str(y_t[0])][:20]))
  im = ax[0,3].imshow(ian)
  ax[0,3].set_title("adversarial+noise: ({}){}\ntrue:({}){}".format(
    y_an.detach().numpy()[0],
    ImageNet_mapping[str(y_an.detach().numpy()[0])][:20],
    y_t[0],
    ImageNet_mapping[str(y_t[0])][:20]))
  
  im = ax[1,0].imshow(no)
  ax[1,0].set_title("blank: Var {}".format(vno))
  im = ax[1,1].imshow(non)
  ax[1,1].set_title("noise: Var {}".format(vnon))
  im = ax[1,2].imshow(na*10)
  ax[1,2].set_title("adversarialx10: Var {}".format(vna))
  im = ax[1,3].imshow(nan)
  ax[1,3].set_title("adversarial+noise: Var {}".format(vnan))
  
  fo = odir+"/"+i_name[:-5]+"summary_plot.png"
  print("generating: {}".format(fo))
  fig.subplots_adjust(left  = 0.125,  # the left side of the subplots of the figure
                      right = 0.9,    # the right side of the subplots of the figure
                      bottom = 0.1,   # the bottom of the subplots of the figure
                      top = 0.9,      # the top of the subplots of the figure
                      wspace = 0.2,   # the amount of width reserved for space between subplots,
                 # expressed as a fraction of the average axis width
                      hspace = 0.2)
  fig.set_size_inches(24,18)
  fig.savefig(fo, dpi=100)
  
