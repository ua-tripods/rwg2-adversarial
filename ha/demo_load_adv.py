import torch
from torchvision import models

# import matplotlib.pyplot as plt
import numpy as np

from glob import glob
import os

os.environ['TORCH_MODEL_ZOO'] = './torchModelZoo/'
model = models.vgg16(pretrained=True)
model.eval()

#change it to your local dir that stores adversarial_examples
baseDirectory = './adversarial_examples' 

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
mean.unsqueeze_(0).unsqueeze_(-1).unsqueeze_(-1)
std.unsqueeze_(0).unsqueeze_(-1).unsqueeze_(-1)

for classDir in glob(baseDirectory + '/*'):
    for fn in glob(classDir + '/*.bin'):
        
        img_np = np.fromfile(fn, dtype=np.float32)#each entry is in [0,1] range
        img_np = img_np.reshape([3,224,224])
        
        img = torch.tensor(img_np)
        img = img.unsqueeze(0) # make it a batch of size 1
        
        img = (img - mean ) / std #normalize it for VGG
        pred = model(img)
        
        argmaxClass = torch.argmax(pred, dim=1)
        trueClass = int(fn.split('_')[-2]) #parse true class from file name
        
        # plt.imshow(img_np.transpose(1,2,0))
        title = 'filename: {}\nargmax class:{}, true class:{}'.format(fn,argmaxClass[0],trueClass)
        # plt.title(title)
        # plt.show()
        print(title)
        break
