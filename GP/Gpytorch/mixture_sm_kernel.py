#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 20:44:25 2020

@author: vr308

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import math
import gpytorch
from gpytorch.constraints import Positive
from gpytorch.kernels import Kernel
from typing import Tuple, Union
from scipy.io import loadmat
#from utils import load_time_series_filenames, load_ts_data


class MixtureSMKernel(Kernel):
    
    has_lengthscale = True
    is_stationary=True

    def __init__(self,   num_mixtures=None,
        ard_num_dims=1,
        batch_shape=torch.Size([]),
        mixture_scales_constraint=None,
        mixture_means_constraint=None,
        mixture_weights_constraint=None,
        alpha_constraint=None, **kwargs):
        super(MixtureSMKernel, self).__init__(ard_num_dims=ard_num_dims, batch_shape=batch_shape, **kwargs)
      
        self.register_parameter(name="raw_alpha", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        if alpha_constraint is None:
            alpha_constraint = Positive()

        self.register_constraint("raw_alpha", alpha_constraint)
        
        if num_mixtures is None:
            raise RuntimeError("num_mixtures is a required argument")

        # This kernel does not use the default lengthscale
        self.num_mixtures = num_mixtures

        if mixture_scales_constraint is None:
            mixture_scales_constraint = Positive()

        if mixture_means_constraint is None:
            mixture_means_constraint = Positive()

        if mixture_weights_constraint is None:
            mixture_weights_constraint = Positive()

        self.register_parameter(
            name="raw_mixture_weights", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, self.num_mixtures))
        )
        ms_shape = torch.Size([*self.batch_shape, self.num_mixtures, 1, self.ard_num_dims])
        self.register_parameter(name="raw_mixture_means", parameter=torch.nn.Parameter(torch.zeros(ms_shape)))
        self.register_parameter(name="raw_mixture_scales", parameter=torch.nn.Parameter(torch.zeros(ms_shape)))

        self.register_constraint("raw_mixture_scales", mixture_scales_constraint)
        self.register_constraint("raw_mixture_means", mixture_means_constraint)
        self.register_constraint("raw_mixture_weights", mixture_weights_constraint)
        
    def initialize_from_data(self, train_x: torch.Tensor, train_y: torch.Tensor, **kwargs):
        """
        Initialize mixture components based on batch statistics of the data.

        :param torch.Tensor train_x: Training inputs
        :param torch.Tensor train_y: Training outputs
        """

        with torch.no_grad():
            if not torch.is_tensor(train_x) or not torch.is_tensor(train_y):
                raise RuntimeError("train_x and train_y should be tensors")
            if train_x.ndimension() == 1:
                train_x = train_x.unsqueeze(-1)
            if self.active_dims is not None:
                train_x = train_x[..., self.active_dims]

            # Compute maximum distance between points in each dimension
            train_x_sort = train_x.sort(dim=-2)[0]
            max_dist = train_x_sort[..., -1, :] - train_x_sort[..., 0, :]

            # Compute the minimum distance between points in each dimension
            dists = train_x_sort[..., 1:, :] - train_x_sort[..., :-1, :]
            # We don't want the minimum distance to be zero, so fill zero values with some large number
            dists = torch.where(dists.eq(0.0), torch.tensor(1.0e10, dtype=train_x.dtype, device=train_x.device), dists)
            sorted_dists = dists.sort(dim=-2)[0]
            min_dist = sorted_dists[..., 0, :]

            # Reshape min_dist and max_dist to match the shape of parameters
            # First add a singleton data dimension (-2) and a dimension for the mixture components (-3)
            min_dist = min_dist.unsqueeze_(-2).unsqueeze_(-3)
            max_dist = max_dist.unsqueeze_(-2).unsqueeze_(-3)
            # Compress any dimensions in min_dist/max_dist that correspond to singletons in the SM parameters
            dim = -3
            while -dim <= min_dist.dim():
                if -dim > self.raw_mixture_scales.dim():
                    min_dist = min_dist.min(dim=dim)[0]
                    max_dist = max_dist.max(dim=dim)[0]
                elif self.raw_mixture_scales.size(dim) == 1:
                    min_dist = min_dist.min(dim=dim, keepdim=True)[0]
                    max_dist = max_dist.max(dim=dim, keepdim=True)[0]
                    dim -= 1
                else:
                    dim -= 1

            # Inverse of lengthscales should be drawn from truncated Gaussian | N(0, max_dist^2) |
            self.mixture_scales = torch.randn_like(self.raw_mixture_scales).mul_(max_dist).abs_().reciprocal_()
            # Draw means from Unif(0, 0.5 / minimum distance between two points)
            self.mixture_means = torch.rand_like(self.raw_mixture_means).mul_(0.5).div(min_dist)
            # Mixture weights should be roughly the stdv of the y values divided by the number of mixtures
            self.mixture_weights = train_y.std().div(self.num_mixtures)
            
    def _create_input_grid(
        self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, last_dim_is_batch: bool = False, **params
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is a helper method for creating a grid of the kernel's inputs.
        Use this helper rather than maually creating a meshgrid.

        The grid dimensions depend on the kernel's evaluation mode.

        :param torch.Tensor x1: ... x n x d
        :param torch.Tensor x2: ... x m x d (for diag mode, these must be the same inputs)
        :param diag: Should the Kernel compute the whole kernel, or just the diag? (Default: True.)
        :type diag: bool, optional
        :param last_dim_is_batch: If this is true, it treats the last dimension
            of the data as another batch dimension.  (Useful for additive
            structure over the dimensions). (Default: False.)
        :type last_dim_is_batch: bool, optional

        :rtype: torch.Tensor, torch.Tensor
        :return: Grid corresponding to x1 and x2. The shape depends on the kernel's mode:
            * `full_covar`: (`... x n x 1 x d` and `... x 1 x m x d`)
            * `full_covar` with `last_dim_is_batch=True`: (`... x k x n x 1 x 1` and `... x k x 1 x m x 1`)
            * `diag`: (`... x n x d` and `... x n x d`)
            * `diag` with `last_dim_is_batch=True`: (`... x k x n x 1` and `... x k x n x 1`)
        """
        x1_, x2_ = x1, x2
        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -2).unsqueeze(-1)
            if torch.equal(x1, x2):
                x2_ = x1_
            else:
                x2_ = x2_.transpose(-1, -2).unsqueeze(-1)

        if diag:
            return x1_, x2_
        else:
            return x1_.unsqueeze(-2), x2_.unsqueeze(-3)
            
    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, last_dim_is_batch: bool = False, **params
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n, num_dims = x1.shape[-2:]

        if not num_dims == self.ard_num_dims:
            raise RuntimeError(
                "The SpectralMixtureKernel expected the input to have {} dimensionality "
                "(based on the ard_num_dims argument). Got {}.".format(self.ard_num_dims, num_dims)
            )
        alpha = self.alpha
        # Expand x1 and x2 to account for the number of mixtures
        # Should make x1/x2 (... x k x n x d) for k mixtures
        x1_ = x1.unsqueeze(-3)
        x2_ = x2.unsqueeze(-3)

        # Compute distances - scaled by appropriate parameters
        x1_exp = x1_ * self.mixture_scales
        x2_exp = x2_ * self.mixture_scales
        x1_cos = x1_ * self.mixture_means
        x2_cos = x2_ * self.mixture_means

        # Create grids
        x1_exp_, x2_exp_ = self._create_input_grid(
            x1_exp, x2_exp, diag=diag, last_dim_is_batch=last_dim_is_batch, **params
        )
        x1_cos_, x2_cos_ = self._create_input_grid(
            x1_cos, x2_cos, diag=diag, last_dim_is_batch=last_dim_is_batch, **params
        )

        # Compute the rq and cosine terms
        rq_term = (1+ (x1_exp_ - x2_exp_).pow_(2).div_(2 * alpha)) 
        cos_term = (x1_cos_ - x2_cos_).mul_(2 * math.pi)
        res = rq_term.pow(-alpha)*cos_term.cos_()

        # Product over dimensions
        if last_dim_is_batch:
            res = res.squeeze(-1)
        else:
            res = res.prod(-1)

        # Sum over mixtures
        mixture_weights = self.mixture_weights.unsqueeze(-1)
        if not diag:
            mixture_weights = mixture_weights.unsqueeze(-1)
        if last_dim_is_batch:
            mixture_weights = mixture_weights.unsqueeze(-1)
            
        #import pdb; pdb.set_trace()
        res = (res * mixture_weights).sum(-2 if diag else -3)
        return res

    
    def forward_rq(self, x1, x2, diag=False, **params):
        def postprocess_rq(dist):
            alpha = self.alpha
            for _ in range(1, len(dist.shape) - len(self.batch_shape)):
                alpha = alpha.unsqueeze(-1)
            return (1 + dist.div(2 * alpha)).pow(-alpha)

        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        return self.covar_dist(
            x1_, x2_, square_dist=True, diag=diag, dist_postprocess_func=postprocess_rq, postprocess=True, **params
        )
    @property
    def mixture_scales(self):
        return self.raw_mixture_scales_constraint.transform(self.raw_mixture_scales)

    @mixture_scales.setter
    def mixture_scales(self, value: Union[torch.Tensor, float]):
        self._set_mixture_scales(value)

    def _set_mixture_scales(self, value: Union[torch.Tensor, float]):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mixture_scales)
        self.initialize(raw_mixture_scales=self.raw_mixture_scales_constraint.inverse_transform(value))

    @property
    def mixture_means(self):
        return self.raw_mixture_means_constraint.transform(self.raw_mixture_means)

    @mixture_means.setter
    def mixture_means(self, value: Union[torch.Tensor, float]):
        self._set_mixture_means(value)

    def _set_mixture_means(self, value: Union[torch.Tensor, float]):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mixture_means)
        self.initialize(raw_mixture_means=self.raw_mixture_means_constraint.inverse_transform(value))

    @property
    def mixture_weights(self):
        return self.raw_mixture_weights_constraint.transform(self.raw_mixture_weights)

    @mixture_weights.setter
    def mixture_weights(self, value: Union[torch.Tensor, float]):
        self._set_mixture_weights(value)

    def _set_mixture_weights(self, value: Union[torch.Tensor, float]):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mixture_weights)
        self.initialize(raw_mixture_weights=self.raw_mixture_weights_constraint.inverse_transform(value))


    @property
    def alpha(self):
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))
        
    
class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_mixtures):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        dist = np.diff(np.sort(train_x))
        dist = dist[np.where(dist != 0)[0]]
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = MixtureSMKernel(num_mixtures=num_mixtures)
        self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        

if __name__== "__main__":

    x = torch.linspace(0.0,5,100)
    #x_long = np.linspace(0,10,200)

    data = loadmat('../data/time-series/' + ts)
    
    #Perform ML-II with multiple restarts 
    print('Performing ML-II with multiple restarts')
    
    import numpy as np
    
    n_restarts = 2
    
    num_steps = 1000

    final_states = []
    losses = torch.zeros(n_restarts, num_steps)
    hyper_means_q1,  hyper_means_q2 = (torch.zeros(n_restarts, num_steps) for i in np.arange(2))
    hyper_scales_q1, hyper_scales_q2 = (torch.zeros(n_restarts, num_steps) for i in np.arange(2))
    hyper_weights_q1,  hyper_weights_q2 = (torch.zeros(n_restarts, num_steps) for i in np.arange(2))
    
    mean_grads_q1, mean_grads_q2 = (torch.zeros(n_restarts, num_steps) for i in np.arange(2))
    scale_grads_q1, scale_grads_q2 = (torch.zeros(n_restarts, num_steps) for i in np.arange(2))

    for i in np.arange(n_restarts):
        
          likelihood = gpytorch.likelihoods.GaussianLikelihood()
          model_ml = SpectralMixtureGPModel(train_x, train_y, likelihood, 2)
         
          # #hypers = {'likelihood.noise_covar.noise': torch.tensor([1.0]),
          #  'mean_module.constant': torch.tensor([0.0]),
          #  'covar_module.mixture_weights': torch.tensor([0.5,0.5]),
          #  'covar_module.mixture_means': torch.tensor([[[1.00]],[[1.00]]]),
          #  'covar_module.mixture_scales': torch.tensor([[[0.5]],[[0.5]]])}
         
          # model_ml.initialize(**hypers)
             
          model_ml.train()
          likelihood.train()
          # Use the adam optimizer
          optimizer = torch.optim.Adam(model_ml.parameters(), lr=0.05)
          # "Loss" for GPs - the marginal log likelihood
          mll_ml = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model_ml)
              
          for j in range(num_steps):
              optimizer.zero_grad()
              output = model_ml(train_x)
              loss = -mll_ml(output, train_y)
              #model_states.append({param_name: param.detach() for param_name, param in model_ml.state_dict().items()})
              #hypers = update_hypers(model_ml, hypers, iter_count)
              hyper_means_q1[i, j] = model_ml.covar_module.mixture_means.detach().flatten()[0]
              hyper_means_q2[i, j] = model_ml.covar_module.mixture_means.detach().flatten()[1]
              hyper_scales_q1[i, j] = model_ml.covar_module.mixture_scales.detach().flatten()[0]
              hyper_scales_q2[i, j] = model_ml.covar_module.mixture_scales.detach().flatten()[1]
              hyper_weights_q1[i, j] = model_ml.covar_module.mixture_weights.detach().flatten()[0]
              hyper_weights_q2[i, j] = model_ml.covar_module.mixture_weights.detach().flatten()[1]
              losses[i, j] = loss.item()
              loss.backward()
             
              mean_grads_q1[i,j] = model_ml.covar_module.raw_mixture_means.grad.detach().flatten()[0]
              mean_grads_q2[i,j] = model_ml.covar_module.raw_mixture_means.grad.detach().flatten()[1]
              scale_grads_q1[i,j] = model_ml.covar_module.raw_mixture_scales.grad.detach().flatten()[0]
              scale_grads_q2[i,j] = model_ml.covar_module.raw_mixture_scales.grad.detach().flatten()[1]

              if (j%100 == 0):
                  print('Iter %d/%d - Loss: %.3f' % (j + 1, num_steps, loss.item()))
              optimizer.step()
         
          print('Preserving final model state for restart iteration ' + str(i))
          final_states.append({param_name: param.detach() for param_name, param in model_ml.state_dict().items()})
        
        
    # Extracting the converged q1 mean and q1 sd for all trajectories for visualisation
    
    raw_mean_q1_trajecs_cnv = [i['covar_module.raw_mixture_means'].detach()[0] for i in final_states]
    raw_scales_q1_trajecs_cnv = [i['covar_module.raw_mixture_scales'].detach()[0] for i in final_states]
    
    mean_q1_trajec_cnv = [model_ml.covar_module.raw_mixture_means_constraint.transform(i).item() for i in raw_mean_q1_trajecs_cnv]
    scales_q1_trajec_cnv = [model_ml.covar_module.raw_mixture_scales_constraint.transform(i).item() for i in raw_scales_q1_trajecs_cnv]

    mse_ = []
    nlpd_ = []
    
    for i in np.arange(n_restarts):
     
        model_params = final_states[i]
        model = SpectralMixtureGPModel(train_x, train_y, likelihood, 2)
        model.load_state_dict(model_params)
        
        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()
    
        dense_x = torch.linspace(-1,1,200, dtype=torch.float64)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Make predictions
            test_pred = likelihood(model(test_x))
            dense_test_pred = likelihood(model(dense_x))
            
        # Metrics on 20 test data points
            
        mse = metrics.mean_squared_error(test_y, test_pred.mean.numpy())
        nlpd = metrics.negative_log_predictive_density(test_y, test_pred.mean, torch.diag(test_pred.covariance_matrix))
       
        print(mse)
        print(nlpd)
       
        mse_.append(mse.item())
        nlpd_.append(nlpd.item())
              
    # Extracting best re-start 
    
    final_idx = torch.argmin(losses[:, -1])
    model_params = final_states[final_idx]
    
    final_model = SpectralMixtureGPModel(train_x, train_y, likelihood, 2)
    final_model.load_state_dict(model_params)
    
    trained_mixture_means = final_model.covar_module.mixture_means.detach()
    trained_mixture_scales = final_model.covar_module.mixture_scales.detach()
    trained_mixture_weights = final_model.covar_module.mixture_weights.detach()
    trained_noise_level = final_model.likelihood.noise_covar.noise.detach().flatten().item()
    lml_value = losses[:,-1][final_idx].item()
    
    
    import matplotlib.pylab as plt
     #  Get into evaluation (predictive posterior) mode
    final_model.eval()
    likelihood.eval()

    dense_x = torch.linspace(0,5,200, dtype=torch.float64)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Make predictions
        test_pred = likelihood(final_model(test_x))
        dense_test_pred = likelihood(final_model(dense_x))

    plt.figure(figsize=(8,3))
    # Get upper and lower confidence bounds
    lower, upper = dense_test_pred.confidence_region()
    plt.fill_between(dense_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5, color='r', label='95% CI')
    plt.plot(train_x.numpy(), train_y.numpy(), color='black', marker='o', linestyle='', markersize=4, label='Observed')
    plt.plot(test_x.numpy(), test_y.numpy(), color='green',  marker='*', linestyle='', markersize=4, label='Test')
    plt.plot(dense_x.numpy(), dense_test_pred.mean.numpy(), color='red', label='Prediction')
    # Shade between the lower and upper confidence bounds
    plt.xticks(fontsize='small')
    plt.yticks(fontsize='small')
    plt.legend(fontsize='small')
    plt.title('ML-II', fontsize='small')
    plt.tight_layout()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    fname = '../results/ml_II_sm_' + str(seed) + '_' + str(n_train) + '_' + str(noise_level) + '.png'
    plt.savefig(fname)