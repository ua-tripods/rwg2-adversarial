cov  = a_sig*np.identity(n_sz)
cov = identity(n_sz)
m = np.zeros([224,224,3])
m = np.zeros([20,20,3])
c = 0.016*np.identity(20*20)
samp = np.random.normal(0,np.sqrt(.016),m.shape)
v_samp = (samp - np.min(samp))/(np.max(samp)-np.min(samp))
#plt.hist(v_samp.flatten())
samp2 = np.random.multivariate_normal(m[:,:,0].flatten(),c,3).swapaxes(0,1).reshape([20,20,3])
v_samp2 = (samp2 - np.min(samp2))/(np.max(samp2)-np.min(samp2))

vv_samp = np.array([np.vstack((v_samp[:,:,0],v_samp2[:,:,0])),
                    np.vstack((v_samp[:,:,1],v_samp2[:,:,1])),
                    np.vstack((v_samp[:,:,2],v_samp2[:,:,2]))]).swapaxes(0,2)
plt.imshow(v_samp)
plt.show()

plt.imshow(vv_samp)
plt.show()
plt.imshow(v_samp2)
plt.show()
