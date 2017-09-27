# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
import numpy as np
import matplotlib.pylab as plt
import random
import seaborn as sns
import scipy.stats as st
import pandas as pd

def gaussian_function(x, mu, var):
    
    normalisation = 1/np.sqrt(2*np.pi*var)
    return normalisation*np.exp(-0.5*np.power((x - mu),2)/var)

x = np.linspace(0,10, 1000)
pdf1 = st.norm.pdf(x, 4, 0.7)
pdf2 = st.norm.pdf(x, 4, 1.0)
unnorm_pdf3 = pdf1*pdf2
scaling_factor = gaussian_function(3, 4, 0.7*0.7+1.5*1.5)
norm_pdf3 = unnorm_pdf3/scaling_factor

#plt.plot(x,norm_pdf3, 'r-')

sns.set_palette(sns.color_palette('hls',8))

mean = [4,4]
cov = [(1,0.5),(0.5,1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])

#plt.figure()
sns.jointplot(x="x", y ="y", data=df, kind = 'kde', title = 'M')
#plt.plot(x, pdf1, 'b-', label = 'N(3, 0.49)')
#plt.plot(x, pdf2, 'g-', label='N(5, 2.25)')
#plt.legend()

fig = plt.figure()
ax = fig.gca(projection='3d')