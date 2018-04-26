# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import scipy.stats as st
from matplotlib import cm
sns.set_palette(sns.color_palette('hls',8))

################ Bi-variate bump ###############################################

# Our 2-dimensional distribution will be over variables X and Y
N = 60
X = np.linspace(-3, 3, N)
Y = np.linspace(-3, 4, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([0., 1.])
Sigma = np.array([[ 1. , -0.5], [-0.5,  1.5]])

z = np.random.multivariate_normal(mu, Sigma, 200)

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu, Sigma)

# Create a surface plot and projected filled contour plot under it.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, antialiased=True,
                cmap=cm.jet, alpha=0.4)
ax.scatter(xs=z[:,0], ys=z[:,1], marker='x', s=20, color='k')

cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.jet)

# Adjust the limits, ticks and view angle
ax.set_zlim(-0.15,0.2)
ax.set_zticks(np.linspace(0,0.2,5))
ax.view_init(27, -21)


######################## Bi-variate Contour ##########################################


def gaussian_function(x, mu, var):
    
    normalisation = 1/np.sqrt(2*np.pi*var)
    return normalisation*np.exp(-0.5*np.power((x - mu),2)/var)

x = np.linspace(0,10, 1000)
pdf1 = st.norm.pdf(x, 4, 0.7)
pdf2 = st.norm.pdf(x, 4, 1.0)
unnorm_pdf3 = pdf1*pdf2
scaling_factor = gaussian_function(3, 4, 0.7*0.7+1.5*1.5)
norm_pdf3 = unnorm_pdf3/scaling_factor