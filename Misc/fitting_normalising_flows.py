#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vr308

Training normalising flows in pytorch

"""

from sklearn import datasets
import torch
from torch import nn
import math
import numpy as np


class NormalisingFlowModel(nn.Module):
       
       def __init__(self, flows):
           super().__init__()
           self.flows = nn.ModuleList(flows)
           
       def forward(self, z):
           m, _ = z.shape
           log_det = torch.zeros(m)
           zk=[z]
           for flow in self.flows:
               z, log_det = flow.forward(z)
               sum_log_det = 
       def get_batch():
           return;
           
       def get_log_prob():
           return;
           
       def train(self, X):
           batch_x = self.get_batch(X)
           
           
           
       
class PlanarTransform():
    #2d planar flow
    def __init__(self, init_sigma=0.01):
        super().__init__()
        self.u = nn.Parameter(torch.randn(1, 2).normal_(0, init_sigma))
        self.w = nn.Parameter(torch.randn(1, 2).normal_(0, init_sigma))
        self.b = nn.Parameter(torch.randn(1).fill_(0))

    def forward(self, z, normalised_u=True):
         # allow for a single forward pass over all the transforms in the flows with a Sequential container
        if isinstance(x, tuple):
            z, sum_log_abs_det_jacobians = x
        else:
            z, sum_log_abs_det_jacobians = x, 0

        # normalize u s.t. w @ u >= -1; sufficient condition for invertibility
        u_hat = self.u
        if normalize_u:
            wtu = (self.w @ self.u.t()).squeeze()
            m_wtu = - 1 + torch.log1p(wtu.exp())
            u_hat = self.u + (m_wtu - wtu) * self.w / (self.w @ self.w.t())

        # compute transform
        f_z = z + u_hat * torch.tanh(z @ self.w.t() + self.b)
        # compute log_abs_det_jacobian
        psi = (1 - torch.tanh(z @ self.w.t() + self.b)**2) @ self.w
        det = 1 + psi @ u_hat.t()
        log_abs_det_jacobian = torch.log(torch.abs(det) + 1e-6).squeeze()
        sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return f_z, sum_log_abs_det_jacobians
        



class RadialTransform():
    #return;
    

def sample_2d_data(dataset, n_samples):

    z = torch.randn(n_samples, 2)

    if dataset == '8gaussians':
        scale = 4
        sq2 = 1/math.sqrt(2)
        centers = [(1,0), (-1,0), (0,1), (0,-1), (sq2,sq2), (-sq2,sq2), (sq2,-sq2), (-sq2,-sq2)]
        centers = torch.tensor([(scale * x, scale * y) for x,y in centers])
        return sq2 * (0.5 * z + centers[torch.randint(len(centers), size=(n_samples,))])

    elif dataset == '2spirals':
        n = torch.sqrt(torch.rand(n_samples // 2)) * 540 * (2 * math.pi) / 360
        d1x = - torch.cos(n) * n + torch.rand(n_samples // 2) * 0.5
        d1y =   torch.sin(n) * n + torch.rand(n_samples // 2) * 0.5
        x = torch.cat([torch.stack([ d1x,  d1y], dim=1),
                       torch.stack([-d1x, -d1y], dim=1)], dim=0) / 3
        return x + 0.1*z

    elif dataset == 'checkerboard':
        x1 = torch.rand(n_samples) * 4 - 2
        x2_ = torch.rand(n_samples) - torch.randint(0, 2, (n_samples,), dtype=torch.float) * 2
        x2 = x2_ + x1.floor() % 2
        return torch.stack([x1, x2], dim=1) * 2

    elif dataset == 'rings':
        n_samples4 = n_samples3 = n_samples2 = n_samples // 4
        n_samples1 = n_samples - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, set endpoint=False in np; here shifted by one
        linspace4 = torch.linspace(0, 2 * math.pi, n_samples4 + 1)[:-1]
        linspace3 = torch.linspace(0, 2 * math.pi, n_samples3 + 1)[:-1]
        linspace2 = torch.linspace(0, 2 * math.pi, n_samples2 + 1)[:-1]
        linspace1 = torch.linspace(0, 2 * math.pi, n_samples1 + 1)[:-1]

        circ4_x = torch.cos(linspace4)
        circ4_y = torch.sin(linspace4)
        circ3_x = torch.cos(linspace4) * 0.75
        circ3_y = torch.sin(linspace3) * 0.75
        circ2_x = torch.cos(linspace2) * 0.5
        circ2_y = torch.sin(linspace2) * 0.5
        circ1_x = torch.cos(linspace1) * 0.25
        circ1_y = torch.sin(linspace1) * 0.25

        x = torch.stack([torch.cat([circ4_x, circ3_x, circ2_x, circ1_x]),
                         torch.cat([circ4_y, circ3_y, circ2_y, circ1_y])], dim=1) * 3.0

        # random sample
        x = x[torch.randint(0, n_samples, size=(n_samples,))]

        # Add noise
        return x + torch.normal(mean=torch.zeros_like(x), std=0.08*torch.ones_like(x))

    else:
        raise RuntimeError('Invalid `dataset` to sample from.')
        
    

if __name__ == '__main__':
    
    z1 = torch.tensor(data=np.linspace(-4,4,200))
    z2 = torch.tensor(data=np.linspace(-4,4,200))
    z1_s, z2_s = torch.meshgrid((z1, z2))
    z_field = torch.stack((z1_s.flatten(), z2_s.flatten()), dim=1).squeeze()
    #z_field = torch.tensor(np.concatenate([z1_s[..., None], z2_s[..., None]], axis=-1)).reshape(10000,2).float()

    
    w1 = lambda z: torch.sin(2 * math.pi * z[:,0] / 4)
    w2 = lambda z: 3 * torch.exp(-0.5 * ((z[:,0] - 1)/0.6)**2)
    w3 = lambda z: 3 * torch.sigmoid((z[:,0] - 1) / 0.3)
    
    u_z1 = lambda z: 0.5 * ((torch.norm(z, p=2, dim=1) - 2) / 0.4)**2 - \
             torch.log(torch.exp(-0.5*((z[:,0] - 2) / 0.6)**2) + torch.exp(-0.5*((z[:,0] + 2) / 0.6)**2) + 1e-10)
    u_z2 = lambda z: 0.5 * ((z[:,1] - w1(z)) / 0.4)**2
    u_z3 = lambda z: - torch.log(torch.exp(-0.5*((z[:,1] - w1(z))/0.35)**2) + torch.exp(-0.5*((z[:,1] - w1(z) + w2(z))/0.35)**2) + 1e-10)
    u_z4 = lambda z: - torch.log(torch.exp(-0.5*((z[:,1] - w1(z))/0.4)**2) + \
                                     torch.exp(-0.5*((z[:,1] - w1(z) + w3(z))/0.35)**2) + 1e-10)


    potential_1 = u_z1(z_field)
    potential_2 = u_z2(z_field)
    potential_3 = u_z3(z_field)
    potential_4 = u_z4(z_field)
    
    # Plotting potential energy functions for density matching
    
    fig, axs = plt.subplots(2,2,figsize=(8,8))
    
    axs[0,0].pcolormesh(z1_s, z2_s, torch.exp(-potential_1).reshape(200,200))
    axs[0,1].pcolormesh(z1_s, z2_s, torch.exp(-potential_2).reshape(200,200))
    axs[1,0].pcolormesh(z1_s, z2_s, torch.exp(-potential_3).reshape(200,200))
    axs[1,1].pcolormesh(z1_s, z2_s, torch.exp(-potential_4).reshape(200,200))
    
    # Plotting samples for density fitting
    
    x_checker = sample_2d_data('checkerboard',1000000)
    x_spiral = sample_2d_data('2spirals',1000000)
    x_rings = sample_2d_data('rings',1000000)
    x_8gauss = sample_2d_data('8gaussians',1000000)
    
    fig, axs = plt.subplots(2,2,figsize=(8,8))
    axs[0,0].hist2d(x_checker[:,0].numpy(), x_checker[:,1].numpy(), bins=1000)
    axs[0,1].hist2d(x_spiral[:,0].numpy(), x_spiral[:,1].numpy(), bins=1000)
    axs[1,0].hist2d(x_rings[:,0].numpy(), x_rings[:,1].numpy(), bins=1000)
    axs[1,1].hist2d(x_8gauss[:,0].numpy(), x_8gauss[:,1].numpy(), bins=1000)


# ----------------------------------------------------
# Training a flow with one of the potential functions
# ----------------------------------------------------

    flow_length = 10
    n_steps = 5000
    batch_size=100
    
    # setup base distribution
    base_dist = D.MultivariateNormal(torch.zeros(2), torch.eye(2))
    
    # set up flow
    flow = nn.Sequential(*[PlanarTransform() for _ in range(flow_length)])

    optimizer = torch.optim.RMSprop(flow.parameters(), lr=args.lr, momentum=0.9, alpha=0.90, eps=1e-6)
    
    for i in range(n_steps):
        
        
        z0 = base_dist.sample(batch_size)
        optimizer.zero_grad()
        loss = flow.log_prob()
















# Fitting a flow - a transformed distribution

n_samples = 1000
X, y = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)

base_dist = dist.Normal(torch.zeros(2), torch.ones(2))
spline_transform = trans.spline_coupling(2, count_bins=16)
spline_transform = trans.planar(2)
flow_dist = dist.TransformedDistribution(base_dist, [spline_transform])

dataset = torch.tensor(X, dtype=torch.float)
optimizer = torch.optim.Adam(spline_transform.parameters(), lr=5e-3)
for step in range(5000):
    optimizer.zero_grad()
    loss = -flow_dist.log_prob(dataset).mean()
    loss.backward()
    optimizer.step()
    flow_dist.clear_cache()

    if step % 500 == 0:
        print('step: {}, loss: {}'.format(step, loss.item()))
        
X_flow = flow_dist.sample(torch.Size([1000,])).detach().numpy()
plt.title(r'Joint Distribution')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.scatter(X[:,0], X[:,1], label='data', alpha=0.5)
plt.scatter(X_flow[:,0], X_flow[:,1], color='firebrick', label='flow', alpha=0.5)
plt.legend()
plt.show()