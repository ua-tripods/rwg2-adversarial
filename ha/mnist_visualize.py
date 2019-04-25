		# mnist_visualize
import numpy as np
import matplotlib.pyplot as plt 
import pickle 
import sys, os
import pandas as pd
import csv

pfile = "C:/Users/Nexus/Google Drive/dropbox/Dropbox/UOFA/0-research/network/rwg2-adversarial/2018-10-01-mnist_attack.pkl" 
#pfile = "C:/Users/DuxLiteratum/Google Drive/dropbox/Dropbox/UOFA/0-research/network/rwg2-adversarial/2018-10-01-mnist_attack.pkl" 
#pfile = "/home/bwbell/Dropbox/UOFA/0-research/network/rwg2-adversarial/2018-10-01-mnist_attack.pkl" 
with open(pfile, "rb") as f: 
    adict = pickle.load(f) 

ox = adict["ox"]
ctrue = adict["ctrue"]
cpred = adict["cpred"]
noises  = adict["noises"]
cpred_a = adict["cpred_a"]  

#visualize N random images 
idxs = np.random.choice(range(500), size=(20,), replace=False)
#idxs = np.arange(0,20)+20
for matidx, idx in enumerate(idxs):
    orig_im = ox[idx].reshape(28,28)
    nse_im = noises[idx].reshape(28,28)
    adv_im  = orig_im + nse_im
    disp_im = np.concatenate((orig_im, adv_im, nse_im), axis=1)
    plt.subplot(5,4,matidx+1)
    plt.imshow(disp_im, "gray")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.title("Orig: {} | New: {} | Adv Pert (variance: {})".format(ctrue[idx], cpred_a[idx], np.sqrt(np.var(nse_im)/np.var(orig_im))))
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

        mnoises = np.mean(anoise[ijclass.tolist()], axis=0).reshape(28,28)
        mx = np.abs(mnoises).max()
        onoises = np.mean(aorigi[ijclass.tolist()], axis=0).reshape(28,28)
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
        orig_im = ox[idx].reshape(28,28)
        nse_im = noises[idx].reshape(28,28)
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
mnoises = np.mean(noises, axis=0).reshape(28,28)

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

plt.hist(aorigi[1]+noises[1],28)
plt.show()
