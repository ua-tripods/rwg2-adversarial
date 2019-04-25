
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
x = np.linspace(0, 5, 10, endpoint=False)
q = multivariate_normal(x, mean=2.5, cov=0.5);
y = multivariate_normal.pdf(x, mean=2.5, cov=0.5);
z = multivariate_normal.cdf(x, mean=2.5, cov=0.5);

plt.plot(x, q)
x = np.linspace(0, 5, 10, endpoint=False)
y = multivariate_normal.pdf(x, mean=2.5, cov=0.5);
plt.plot(x, y)

x, y = np.mgrid[-1:1:.01, -1:1:.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
n_sz = 50176
mean = np.zeros(n_sz)
cov = identity(n_sz)
rv = multivariate_normal(mean, cov)#[0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
plt.contourf(x, y, rv.pdf(pos))
