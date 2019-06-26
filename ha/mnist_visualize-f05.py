import numpy as np
import matplotlib.pyplot as plt 
import pickle 
import sys, os
import pandas as pd
import csv
		# mnist_visualize

#pfile = "C:/Users/Nexus/Google Drive/dropbox/Dropbox/UOFA/0-research/network/rwg2-adversarial/2018-10-01-mnist_attack.pkl" 

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
    ddir = "~/Adversarial-Examples-in-PyTorch/mnist_scale"
    os.makedirs(ddir, exist_ok=True)
    mdir = "~/Adversarial-Examples-in-PyTorch/mnist2"
    os.makedirs(mdir, exist_ok=True)
    odir = "~/Adversarial-Examples-in-Pytorch/imnet_examples"
    os.makedirs(odir, exist_ok=True)

fo    = ddir + "/Config-2-weights.pkl"
wfile = ddir + "/Config-2-weights.pkl"
efile = ddir + "/mnist_examples.pkl"

pfile = ddir+"mnist_attack-L2Loss-f05.pkl"
sf = 2
sl = 28
sw = 28
sli = int(sl/sf)
swi = int(sw/sf)

#pfile = "C:/Users/DuxLiteratum/Google Drive/dropbox/Dropbox/UOFA/0-research/network/rwg2-adversarial/2018-10-01-mnist_attack.pkl" 
#pfile = "/home/bwbell/Dropbox/UOFA/0-research/network/rwg2-adversarial/2018-10-01-mnist_attack.pkl" 
with open(pfile, "rb") as f: 
    adict = pickle.load(f) 

ox = adict["ox"]
noises  = adict["noises"]
ctrue = adict["ctrue"]
cpred = adict["cpred"]
cpred_a = adict["cpred_a"]  

#visualize N random images 
#idxs = np.random.choice(range(len(ox)), size=(20,), replace=False)
#idx = np.arange(0,20)+20
idx = 0
matidx = 0
orig_im = ox[idx].reshape(sli,swi)
for i in range(0,len(noises[idx])):
    noi = noises[idx][i]
    ct  = ctrue[idx][i]
    cp  = cpred[idx][i]
    ca  = cpred_a[idx][i]
    nse_im = noi.reshape(sli,swi)
    adv_im  = orig_im + nse_im
    disp_im = np.concatenate((orig_im, adv_im, nse_im), axis=1)
    plt.subplot(3,3,matidx+1)
    matidx += 1
    plt.imshow(disp_im, "gray")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.title("Orig: {} | New: {} | Var: {:.2f})".format(ct, ca, np.sqrt(np.var(nse_im)/np.var(orig_im))))
plt.show()


# plot noise for cartesian product
#      partition sampels by the 10 classes
iclass = []
anoise = np.array(noises)
aorigi = np.array(ox)

for i in list(range(0,10)):
    for j in list(range(0,10)):
        #    find adversarial examples that take i to j
        ijclass = np.intersect1d(np.where(np.array(ctrue) == i),
                               np.where(np.array(cpred_a) == j))
        if (i == j):
            continue

        mnoises = np.mean(anoise[ijclass.tolist()], axis=0).reshape(sli,swi)
        mx = np.abs(mnoises).max()
        onoises = np.mean(aorigi[ijclass.tolist()], axis=0).reshape(sli,swi)
        plt.subplot(10,10,10*i+j+1)
	# try Carlos's colors
        plt.imshow(mnoises, vmin=-mx, vmax=mx, cmap="RdBu")
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title("{} : {},{:.2f}".format(i,j, np.sqrt(np.var(mnoises)/np.var(onoises)))) 
        print("Plotted {} : {}".format(i,j))
plt.show()

for i in list(range(0,10)):
    for j in list(range(0,10)):
        #    find adversarial examples that take i to j
        ijclass = np.intersect1d(np.where(np.array(ctrue) == i),
                               np.where(np.array(cpred_a) == j))
        if (i == j):
            continue

        mmnoises = np.sqrt(np.var(anoise[ijclass.tolist()],1)/np.var(aorigi[ijclass.tolist()],1))
        plt.subplot(10,10,10*i+j+1)
        plt.hist(mmnoises,24)
        print("Plotted {} : {}".format(i,j))
plt.show()

for i in list(range(0,10)):
    for j in list(range(0,10)):
        #    find adversarial examples that take i to j
        ijclass = np.intersect1d(np.where(np.array(ctrue) == i),
                               np.where(np.array(cpred_a) == j))
        if (i == j):
            continue

        mmnoises = np.sqrt(np.var(anoise[ijclass.tolist()],1)/np.var(aorigi[ijclass.tolist()],1))
        idx = ijclass[np.argmin(mmnoises)]
        orig_im = ox[idx].reshape(sli,swi)
        nse_im = noises[idx].reshape(sli,swi)
        adv_im  = orig_im + nse_im
        disp_im = np.concatenate((orig_im, adv_im, nse_im), axis=1)
        plt.subplot(10,10,10*i+j+1)
        plt.imshow(disp_im, vmin=-mx, vmax=mx, cmap="RdBu")
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title("{} | {} | Var: {:.2f})".format(ctrue[idx], cpred_a[idx], np.sqrt(np.var(nse_im)/np.var(orig_im))))


        plt.hist(mmnoises,24)
        print("Plotted {} : {}".format(i,j))
plt.show()



# plot mean noise for everybody
mnoises = np.mean(noises, axis=0).reshape(sli,swi)

# compute pointwise noise variances and then draw histogram
mmnoises = np.sqrt(np.var(anoise,1)/np.var(aorigi,1))
plt.hist(mmnoises,144)
plt.show()

mx = np.abs(mnoises).max()
plt.imshow(mnoises, vmin=-mx, vmax=mx, cmap="RdBu")
plt.xticks([])
plt.yticks([])
plt.title("Mean Noise")
plt.colorbar()
plt.show()

# Noise statistics 
noises, ox, ctrue, cpred = np.array(noises), np.array(ox), np.array(ctrue), np.array(cpred)
adv_exs = ox + noises
print("Adv examples: max, min: ", adv_exs.max(), adv_exs.min())
print("Noise: Mean, Max, Min: ")
print(np.mean(noises), np.max(noises), np.min(noises))

# output .csv of dataset
indices = np.arange(1,9556)
fo = "2018-10-03-Adversarial_Noises.csv"
anoises = np.append(indices[...,None],anoise,1)
with open(fo,"w") as foc:
  cw = csv.writer(foc,delimiter=',')
  cw.writerows(anoises)

fo = "2018-10-03-Adversarial_Images.csv"
aorigis = np.append(indices[...,None],aorigi,1)
with open(fo,"w") as foc:
  cw = csv.writer(foc,delimiter=',')
  cw.writerows(aorigi)

plt.hist(aorigi[1]+noises[1],sli)
plt.show()
