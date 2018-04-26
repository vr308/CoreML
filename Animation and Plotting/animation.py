#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 00:55:33 2018

@author: vr308
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0,10), ylim=(-10,10))
line, = ax.plot([], [], lw=2)
xdata, ydata = [], []

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

def data_gen(x=0):
    cnt = 0
    while cnt < 100:
        cnt += 1
        x += 0.1
        yield x, x*np.sin(x)
    
def animate(data):
    x, y = data
    xdata.append(x)
    ydata.append(y)
    line.set_data(xdata, ydata)
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, func=animate, frames=data_gen, init_func=init,
           repeat= True, interval=500, blit=True)

plt.show()


import matplotlib.pyplot as plt
import numpy  as np
import time

class DynamicUpdate():
    #Suppose we know the x range
    min_x = -2
    max_x = 4

    def on_launch(self):
        #Set up plot
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([],[], 'b-')
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(self.min_x, self.max_x)
        #Other stuff
        self.ax.grid()

    def on_running(self, xdata, ydata):
        #Update data (with the new _and_ the old points)
        self.lines.set_xdata(xdata)
        self.lines.set_ydata(ydata)
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    #Example
    def __call__(self):
      
        self.on_launch()
        xdata = []
        ydata = []
        for x in np.linspace(-2,4,100):
            xdata.append(x)
            ydata.append(np.multiply(np.sin(np.power(x,2)),x))
            self.on_running(xdata, ydata)
            time.sleep(0.1)
        return xdata, ydata

d = DynamicUpdate()
d()


import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.animation as animation

fig, ax = plt.subplots()

# histogram our data with numpy
data = np.random.randn(1000)
n, bins = np.histogram(data, 100)

# get the corners of the rectangles for the histogram
left = np.array(bins[:-1])
right = np.array(bins[1:])
bottom = np.zeros(len(left))
top = bottom + n
nrects = len(left)

# here comes the tricky part -- we have to set up the vertex and path
# codes arrays using moveto, lineto and closepoly

# for each rect: 1 for the MOVETO, 3 for the LINETO, 1 for the
# CLOSEPOLY; the vert for the closepoly is ignored but we still need
# it to keep the codes aligned with the vertices
nverts = nrects*(1 + 3 + 1)
verts = np.zeros((nverts, 2))
codes = np.ones(nverts, int) * path.Path.LINETO
codes[0::5] = path.Path.MOVETO
codes[4::5] = path.Path.CLOSEPOLY
verts[0::5, 0] = left
verts[0::5, 1] = bottom
verts[1::5, 0] = left
verts[1::5, 1] = top
verts[2::5, 0] = right
verts[2::5, 1] = top
verts[3::5, 0] = right
verts[3::5, 1] = bottom

barpath = path.Path(verts, codes)
patch = patches.PathPatch(
    barpath, facecolor='green', edgecolor='yellow', alpha=0.5)
ax.add_patch(patch)

ax.set_xlim(left[0], right[-1])
ax.set_ylim(bottom.min(), top.max())


def animate(i):
    # simulate new data coming in
    data = np.random.randn(1000)
    n, bins = np.histogram(data, 100)
    top = bottom + n
    verts[1::5, 1] = top
    verts[2::5, 1] = top
    return [patch, ]

ani = animation.FuncAnimation(fig, animate, 100, repeat=False, blit=True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = gcf()
ax = gca()
x = np.linspace(0, 2*np.pi, 256)
line, ( bottoms, tops), verts =  ax.errorbar(x, np.sin(x), yerr=1)

verts[0].remove() # remove the vertical lines

yerr = 1
def animate(i=0):
    #    ax.errorbar(x, np.array(x), yerr=1, color='green')
    y = np.sin(x+i/10.0)
    line.set_ydata(y)  # update the data
    bottoms.set_ydata(y - yerr)
    tops.set_ydata(y + yerr)
    return line, bottoms, tops


def init():
    # make an empty frame
    line.set_ydata(np.nan * np.ones(len(line.get_xdata())))
    bottoms.set_ydata(np.nan * np.ones(len(line.get_xdata())))
    tops.set_ydata(np.nan * np.ones(len(line.get_xdata())))
    return line, bottoms, tops


ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,
    interval=25, blit=True)
plt.show()


x = np.linspace(0, 1)
y = np.sin(4 * np.pi * x) * np.exp(-5 * x) * 120

fig, ax = plt.subplots()

# plot only the outline of the polygon, and capture the result
poly, = ax.fill(x, y, facecolor='none')

# get the extent of the axes
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

# create a dummy image
img_data = np.arange(ymin,ymax,(ymax-ymin)/100.)
img_data = img_data.reshape(img_data.size,1)

# plot and clip the image
im = ax.imshow(img_data, aspect='auto', origin='lower', cmap=plt.cm.Reds_r, extent=[xmin,xmax,ymin,ymax], vmin=y.min(), vmax=30.)

im.set_clip_path(poly)


###############################################


plt.figure()

ax = plt.axes(xlim=(0,20), ylim=(-40,40))
poly, = ax.fill(x_array, y_lower, 'b', x_array, y_upper, 'r')

# get the extent of the axes
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

img_data = np.arange()


z1 = [[z[i] for i in np.arange(1000)] for j in np.arange(1000)]

CS = plt.contourf(x_array, y_upper, z1, 200, # \[-1, -0.1, 0, 0.1\],
                        cmap=plt.cm.viridis)



from scipy import interpolate

def draw_tangent(x,y,a):
 # interpolate the data with a spline
 spl = interpolate.splrep(x,y)
 small_t = np.arange(a-1,a+2)
 fa = interpolate.splev(a,spl,der=0)     # f(a)
 fprime = interpolate.splev(a,spl,der=1) # f'(a)
 tan = fa+fprime*(small_t-a) # tangent
 plt.plot(a,fa,'om',small_t,tan,'--r')
 

plt.figure()
plt.plot(x_array[id], y_lower[id])
plt.plot(x_array[id], y_upper[id])
 
draw_tangent(x_array, y_lower, 17)
draw_tangent(x_array, y_upper, 17)

upper_coords = np.column_stack((x_array, y_upper))
lower_coords = np.column_stack((x_array, y_lower))

dist_matrix = distance.cdist(upper_coords, lower_coords, metric='euclidean')

z=[np.min(dist_matrix[i]) for i in np.arange(len(upper_coords))]

plt.plot(x_array, y_upper)
plt.plot(x_array, y_lower)

plt.figure()
plt.plot(x_array, z)
