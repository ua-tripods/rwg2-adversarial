		# mnist_visualize
import numpy as np
import matplotlib.pyplot as plt 
import pickle 
import sys, os 

pfile = "mnist_attack.pkl" 
with open(pfile, "rb") as f: 
    adict = pickle.load(f) 

ox = adict["ox"]
ctrue = adict["ctrue"]
cpred = adict["cpred"]
noises  = adict["noises"]
cpred_a = adict["cpred_a"]  

# visualize N random images 
# idxs = np.random.choice(range(500), size=(20,), replace=False)
# #idxs = np.arange(0,20)+20
# for matidx, idx in enumerate(idxs):
#     orig_im = ox[idx].reshape(28,28)
#     nse_im = noises[idx].reshape(28,28)
#     adv_im  = orig_im + nse_im
#     disp_im = np.concatenate((orig_im, adv_im, nse_im), axis=1)
#     plt.subplot(5,4,matidx+1)
#     plt.imshow(disp_im, "gray")
#     plt.xticks([])
#     plt.yticks([])
#     plt.title("Orig: {} | New: {} | Adv Pert (variance: {})".format(ctrue[idx], cpred_a[idx], np.sqrt(np.var(nse_im)/np.var(orig_im))))
# plt.show()


# plot noise for all transforms
# partition sampels by the 10 classes
iclass = []
anoise = np.array(noises)
aorigi = np.array(ox)
for i in list(range(0,10)):
    for j in list(range(0,10)):
        #    find adversarial examples that take i to j
        ijclass = np.intersect1d(np.where(np.array(ctrue) == i),
                               np.where(np.array(cpred_a) == j))
        if (i >= j):
            continue
        # if (ijclass.__len__() == 0):
        #     mnoises = np.zeros(28*28).reshape(28,28)
        #     onoises = np.ones(28*28).reshape(28,28)

        mnoises = np.mean(anoise[ijclass.tolist()], axis=0).reshape(28,28)
        onoises = np.mean(aorigi[ijclass.tolist()], axis=0).reshape(28,28)
        plt.subplot(9,9,9*i+j)
        plt.imshow(mnoises, "gray")
        plt.xticks([])
        plt.yticks([])
        plt.title("{} : {},{:.2f}".format(i,j, np.sqrt(np.var(mnoises)/np.var(onoises)))) 
        print("Plotted {} : {}".format(i,j))
# Plot the cartesian product of transforms
plt.show()

# plot just the noise
# TODO plot noise average from each class to each other
mnoises = np.mean(noises, axis=0).reshape(28,28)
plt.imshow(mnoises, "gray")
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
