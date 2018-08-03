# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import scipy.stats as st 
import sklearn as skt
import matplotlib.pylab as plt
from statsmodels.nonparametric.kde import KDEUnivariate

c1 = st.cauchy.rvs(loc=-1,scale=0.2, size=10)
c2 = st.cauchy.rvs(loc=0.8,scale=0.2, size=10)

plt.figure()
plt.hist(c1, bins=50)
plt.hist(c2, bins=50)

# Mixture of 2 Cauchy dist. 
c = np.ravel(np.vstack((c1,c2)))
plt.hist(c, bins=50, density=True)

kde = KDEUnivariate(c)
kde.fit(bw=0.2)

x_grid = np.linspace(-4,10, 10000)
plt.plot(x_grid, kde.evaluate(x_grid))

plt.title('Mixture of Cauchy\'s and KDE estimate')




