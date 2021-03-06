import numpy as np
import matplotlib.pyplot as plt 
import pickle 
import sys, os
import pandas as pd
import csv
from   tqdm import *
		# mnist_visualize

def n_mag(ni, oi, method):
  if method == "l2":
    print("not written yet")
    # replace variance calcuoation with sqrt(||noise||^2/||original||^2)
    return np.sqrt(np.linalg.norm(ni)/np.linalg.norm(oi))
  else:
    return np.sqrt(np.var(ni)/np.var(oi))

#afile = "C:/Users/Nexus/Google Drive/dropbox/Dropbox/UOFA/0-research/network/rwg2-adversarial/2018-10-01-mnist_attack.pkl" 

# Sort out Directories
sfile = "C:/Users/Nexus/Google Drive/dropbox/Dropbox/UOFA/0-research/network/rwg2-adversarial/ha/imnet_fgsm.py"
lfile = "/home/bwbell/Dropbox/UOFA/0-research/network/rwg2-adversarial/ha/imnet_fgsm.py"
if (os.path.isfile(sfile)):
    ddir = "C:/Users/Nexus/Desktop/Adversarial-Examples-in-PyTorch2/mnist_scale"
    idir = "C:/Users/Nexus/Desktop/Adversarial-Examples-in-PyTorch2/mnist_scale/img"

    odir = "C:/Users/Nexus/Google Drive/dropbox/Dropbox/UOFA/0-research/network/imnet_examples_nexus"
    ldir = "C:/Users/Nexus/Desktop/Adversarial-Examples-in-PyTorch2/ilsvrc12_imnet_labels"
elif (os.path.isfile(lfile)):
    ddir = "/home/bwbell/Desktop/Adversarial-Examples-in-PyTorch/mnist_scale"
    idir = "/home/bwbell/Desktop/Adversarial-Examples-in-PyTorch/mnist_scale/img"
    mdir = "/home/bwbell/Desktop/Adversarial-Examples-in-PyTorch/mnist2"
    odir = "/home/bwbell/Desktop/Adversarial-Examples-in-PyTorch/mnist_scale"
    # directory for labels in ilsvrc_12
    ldir = "/home/bwbell/Desktop/Adversarial-Examples-in-PyTorch/ilsvrc12_imnet_labels"
elif (os.path.isfile(lfile)):
    ddir = "/home/bwbell/Adversarial-Examples-in-PyTorch/mnist_scale"
    idir = "/home/bwbell/Adversarial-Examples-in-PyTorch/mnist_scale/img"
    mdir = "/home/bwbell/Adversarial-Examples-in-PyTorch/mnist2"
    odir = "/home/bwbell/Adversarial-Examples-in-Pytorch/mnist_scale"
else:
    ddir = "/home/bwbell/Adversarial-Examples-in-PyTorch/mnist_scale"
    os.makedirs(ddir, exist_ok=True)
    idir = "/home/bwbell/Adversarial-Examples-in-PyTorch/mnist_scale/img"
    os.makedirs(idir, exist_ok=True)
    mdir = "/home/bwbell/Adversarial-Examples-in-PyTorch/mnist2"
    os.makedirs(mdir, exist_ok=True)
    odir = "/home/bwbell/Adversarial-Examples-in-PyTorch/mnist_scale"
    os.makedirs(odir, exist_ok=True)


sf = 1
if (len(sys.argv) > 1):
  sf = np.float(sys.argv[1])
cw = "s1"
if (len(sys.argv) > 2):
  cw = sys.argv[2]
fileheader = "f{}-{}".format(sf, cw)
print("File Header: {}".format(fileheader))
print("Running With Scaling Factor {} (1/sf)".format(1/sf))

sl = 28
sw = 28
sli = int(sl/sf)
swi = int(sw/sf)

fo    = ddir + "/Config-2-weights.pkl"
wfile = ddir + "/Config-2-weights-"+fileheader+".pkl"
efile = ddir + "/mnist_examples-"+fileheader+".pkl"
afile = ddir+"mnist_attack-L2Loss-"+fileheader+".pkl"
wfile = ddir + "/Config-2-weights-"+fileheader+".npy"
efile = ddir + "/mnist_examples-"+fileheader+".npy"
afile = ddir+"mnist_attack-L2Loss-"+fileheader+".npy"

#afile = "C:/Users/DuxLiteratum/Google Drive/dropbox/Dropbox/UOFA/0-research/network/rwg2-adversarial/2018-10-01-mnist_attack.pkl" 
#afile = "/home/bwbell/Dropbox/UOFA/0-research/network/rwg2-adversarial/2018-10-01-mnist_attack.pkl" 
# with open(afile, "rb") as f: 
#     adict = pickle.load(f) 
adict = np.load(afile).item()
print("finished loading adversarial examples from: {}".format(afile))

ox = adict["ox"]
noises  = adict["noises"]
ctrue = adict["ctrue"]
cpred = adict["cpred"]
cpred_a = adict["cpred_a"]  

#visualize N random images 
#idxs = np.random.choice(range(len(ox)), size=(20,), replace=False)
#idx = np.arange(0,20)+20
n_min = np.float(0.0)
n_max = np.float(1.0)
nse_ind = np.zeros([10,10])
nse_var = np.zeros([10,10,5120])
nse_all = np.array([])
for i in range(0,len(noises.values())):
  if (len(noises[i]) > 0):
    noi = list(noises.values())[i]# k, noi in noises.values():
    o   = ox[i]
    cpa = list(cpred_a.values())[i]
    ctr = int(list(ctrue.values())[i])
    for j in range(0,len(noi)):
      n  = noi[j].detach().numpy()
      ca = int(cpa[j])

      nse_var[ctr, ca, int(nse_ind[ctr,ca])] = n_mag(n,o,"sqrt")
      nse_ind[ctr, ca]+=1
      nse_all = np.append(nse_all, n_mag(n,o,"sqrt"))
      n_min = np.min([n_min, np.min(n)])
      n_max = np.max([n_max, np.max(n)])
    # while we're here, grab 
    a_max = np.max(np.abs([n_min, n_max]))
    a_min = np.min(np.abs([n_min, n_max]))

# X = np.random.rand(100, 1000)
# loop through data
# hardcode 100 buckets from 0 to amax
buckets = np.arange(0,a_max, a_max/100)
counts= np.zeros(len(buckets))
xs = np.zeros(len(nse_all))
ys = np.zeros(len(nse_all))
adv_ims = np.zeros([len(nse_all),sli,swi])
count = 0
print("Starting to build histogram")
for idx in range(0,len(ox)):
  matidx = 0
  if (idx % 50 == 0):
    print("Finished dumping through {}/{}".format(idx, len(ox)))

  orig_im = ox[idx].reshape(sli,swi)
  if (len(noises[idx]) > 0):
    ct  = int(ctrue[idx])
    cp  = int(cpred[idx])
    for i in range(0,len(noises[idx])):
      noi = noises[idx][i]
      ca  = cpred_a[idx][i]
      nse_im = noi.detach().numpy().reshape(sli,swi)
      adv_im  = orig_im + nse_im
      disp_im = np.concatenate((orig_im, adv_im, nse_im), axis=1)
      nse_va = n_mag(nse_im, orig_im, "sqrt")
      # for each datapoint find its bucket
      adv_ims[count] = adv_im

      ind = sum(buckets < nse_va) - 1
      bucket = buckets[ind]
      #, make its x-coordinate its bucket 
      xs[count] = bucket
      # and its y coordinate the old y for that bucket --
      #print("ys[count] = counts[ind]: {}[{}] = {}[{}]".format(ys, count, counts, ind))
      ys[count] = counts[ind]
      # increment the bucket
      counts[ind]+=1
      count+=1

      



# replace xs and ys with positions for generated histogram

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('click on point to show image: {} - Mean: {}'.format(fileheader, np.mean(nse_all)))
line, = ax.plot(xs, ys, 'o', picker=1)  # 5 points tolerance


def onpick(event):

    if event.artist!=line: return True

    N = len(event.ind)
    if not N: return True


    figi = plt.figure()
    for subplotnum, dataind in enumerate(event.ind):
      print("ind: {}".format(dataind))
      ax = figi.add_subplot(N,1,subplotnum+1)
      ax.imshow(adv_ims[dataind])
      ax.text(0.05, 0.9, 'noise=%1.3f'%(xs[dataind]),
                transform=ax.transAxes, va='top')
        #ax.set_ylim(-0.5, 1.5)
    figi.show()
    return True

fig.canvas.mpl_connect('pick_event', onpick)

plt.show()




# plot the whole histogram
fig1 = plt.figure()
print("total: {}".format(len(nse_all)))
print(nse_all)
plt.hist(nse_all, 100)
fig1.set_size_inches(18, 9)
fo = ddir+"/"+fileheader+"-hist-all.png"
plt.title("Overall Histogram: {} -  Mean: {}".format(fileheader, np.mean(nse_all)))
fig1.savefig(fo, dpi=100)  
print("Saved all histogram to {}".format(fo))
# keep these for reference later
hsave = {}
hsave["all"] = nse_all
hsave["each"] = nse_var
hsave["count"] = nse_ind
fo = ddir+"/mnist_histograms-"+fileheader+".pkl"
with open(fo,"wb") as f: 
    pickle.dump(hsave, f) 
print("Dumped Histograms to: {}".format(fo)) 


# plot each histogram
fig2= plt.figure()
for i in list(range(0,10)):
    for j in list(range(0,10)):
        #    find adversarial examples that take i to j
        if (nse_ind[i,j] > 0):
          plt.subplot(10,10,10*i+j+1)
          plt.hist(nse_var[i,j,range(0,int(nse_ind[i,j])+1)], 24, range=(np.min(nse_all), np.max(nse_all))) 
          print("Plotted {} : {}".format(i,j))
fig2.set_size_inches(30, 15)
fo = ddir+"/"+fileheader+"-hist-each.png"
fig2.savefig(fo, dpi=100)  
print("Saved each histogram to {}".format(fo))

# dump a bunch of examples

if (False):
  for idx in range(0,len(ox)):
    matidx = 0
    if (idx % 50 == 0):
      print("Finished dumping through {}/{}".format(idx, len(ox)))

    orig_im = ox[idx].reshape(sli,swi)
    if (len(noises[idx]) > 0):
      ct  = int(ctrue[idx])
      cp  = int(cpred[idx])
      fig= plt.figure()
      for i in range(0,len(noises[idx])):
        noi = noises[idx][i]
        ca  = cpred_a[idx][i]
        nse_im = noi.detach().numpy().reshape(sli,swi)
        adv_im  = orig_im + nse_im
        disp_im = np.concatenate((orig_im, adv_im, nse_im), axis=1)
        plt.subplot(3,3,matidx+1)
        matidx += 1
        plt.imshow(disp_im, vmin=-a_max, vmax=a_max, cmap="RdBu")
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title("Orig: {} | New: {} | Var: {:.2f})".format(ct, ca, n_mag(nse_im, orig_im, "sqrt")))

      fig.set_size_inches(18, 9)
      fo = idir+"/"+fileheader+"-FGSM-{}-{}.png".format(idx, int(ctrue[idx]))
      fig.savefig(fo, dpi=100)  
 
# display interactive histogram









# # plot noise for cartesian product
# #      partition sampels by the 10 classes
# iclass = []
# anoise = np.array(noises)
# aorigi = np.array(ox)

# for i in list(range(0,10)):
#     for j in list(range(0,10)):
#         #    find adversarial examples that take i to j
#         ijclass = np.intersect1d(np.where(np.array(ctrue) == i),
#                                np.where(np.array(cpred_a) == j))
#         if (i == j):
#             continue

#         mnoises = np.mean(anoise[ijclass.tolist()], axis=0).reshape(sli,swi)
#         mx = np.abs(mnoises).max()
#         onoises = np.mean(aorigi[ijclass.tolist()], axis=0).reshape(sli,swi)
#         plt.subplot(10,10,10*i+j+1)
# 	# try Carlos's colors
#         plt.imshow(mnoises, vmin=-mx, vmax=mx, cmap="RdBu")
#         plt.xticks([])
#         plt.yticks([])
#         plt.colorbar()
#         plt.title("{} : {},{:.2f}".format(i,j, np.sqrt(np.var(mnoises)/np.var(onoises)))) 
#         print("Plotted {} : {}".format(i,j))
# plt.show()

# for i in list(range(0,10)):
#     for j in list(range(0,10)):
#         #    find adversarial examples that take i to j
#         ijclass = np.intersect1d(np.where(np.array(ctrue) == i),
#                                np.where(np.array(cpred_a) == j))
#         if (i == j):
#             continue

#         mmnoises = np.sqrt(np.var(anoise[ijclass.tolist()],1)/np.var(aorigi[ijclass.tolist()],1))
#         plt.subplot(10,10,10*i+j+1)
#         plt.hist(mmnoises,24)
#         print("Plotted {} : {}".format(i,j))
# plt.show()

# for i in list(range(0,10)):
#     for j in list(range(0,10)):
#         #    find adversarial examples that take i to j
#         ijclass = np.intersect1d(np.where(np.array(ctrue) == i),
#                                np.where(np.array(cpred_a) == j))
#         if (i == j):
#             continue

#         mmnoises = np.sqrt(np.var(anoise[ijclass.tolist()],1)/np.var(aorigi[ijclass.tolist()],1))
#         idx = ijclass[np.argmin(mmnoises)]
#         orig_im = ox[idx].reshape(sli,swi)
#         nse_im = noises[idx].reshape(sli,swi)
#         adv_im  = orig_im + nse_im
#         disp_im = np.concatenate((orig_im, adv_im, nse_im), axis=1)
#         plt.subplot(10,10,10*i+j+1)
#         plt.imshow(disp_im, vmin=-mx, vmax=mx, cmap="RdBu")
#         plt.xticks([])
#         plt.yticks([])
#         plt.colorbar()
#         plt.title("{} | {} | Var: {:.2f})".format(ctrue[idx], cpred_a[idx], np.sqrt(np.var(nse_im)/np.var(orig_im))))


#         plt.hist(mmnoises,24)
#         print("Plotted {} : {}".format(i,j))
# plt.show()



# # plot mean noise for everybody
# mnoises = np.mean(noises, axis=0).reshape(sli,swi)

# # compute pointwise noise variances and then draw histogram
# mmnoises = np.sqrt(np.var(anoise,1)/np.var(aorigi,1))
# plt.hist(mmnoises,144)
# plt.show()

# mx = np.abs(mnoises).max()
# plt.imshow(mnoises, vmin=-mx, vmax=mx, cmap="RdBu")
# plt.xticks([])
# plt.yticks([])
# plt.title("Mean Noise")
# plt.colorbar()
# plt.show()

# # Noise statistics 
# noises, ox, ctrue, cpred = np.array(noises), np.array(ox), np.array(ctrue), np.array(cpred)
# adv_exs = ox + noises
# print("Adv examples: max, min: ", adv_exs.max(), adv_exs.min())
# print("Noise: Mean, Max, Min: ")
# print(np.mean(noises), np.max(noises), np.min(noises))

# # output .csv of dataset
# indices = np.arange(1,9556)
# fo = "2018-10-03-Adversarial_Noises.csv"
# anoises = np.append(indices[...,None],anoise,1)
# with open(fo,"w") as foc:
#   cw = csv.writer(foc,delimiter=',')
#   cw.writerows(anoises)

# fo = "2018-10-03-Adversarial_Images.csv"
# aorigis = np.append(indices[...,None],aorigi,1)
# with open(fo,"w") as foc:
#   cw = csv.writer(foc,delimiter=',')
#   cw.writerows(aorigi)

# plt.hist(aorigi[1]+noises[1],sli)
# plt.show()

