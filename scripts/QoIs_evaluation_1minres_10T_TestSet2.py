#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import tqdm
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel, PeriodicKernel, MaternKernel, SpectralDeltaKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood
import pandas as pd
from sklearn.preprocessing import StandardScaler
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL
import numpy as np
from matplotlib import pyplot as plt
import os
import math
import scipy
from rich.progress import track
import datetime
from sklearn.metrics import mean_absolute_error
import torch.nn as nn
import pickle
from torch.utils.data import TensorDataset, DataLoader, Dataset
import re
import matplotlib.dates as mdates


# In[ ]:


def get_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def df_to_tensor(x: pd.DataFrame) -> torch:
    return torch.from_numpy(x.values).to(get_device())

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_hss_count(true_values, predictions, threshold):
    """
    Compute Heidke Skill Score based on a given threshold.
    
    Parameters:
    - true_values: List or array of true values.
    - predictions: List or array of predicted values.
    - threshold: Threshold value to determine event occurrence.
    
    Returns:
    - HSS: Heidke Skill Score.
    """

    # Convert numerical values to binary based on threshold
    true_binary = pd.DataFrame((true_values >= threshold).astype(int))
    pred_binary = pd.DataFrame((predictions >= threshold).astype(int))
    pred_binary.index = true_binary.index

    # Calculate confusion matrix values
    a = np.sum((true_binary == 1) & (pred_binary == 1))
    b = np.sum((true_binary == 0) & (pred_binary == 1))
    c = np.sum((true_binary == 1) & (pred_binary == 0))
    d = np.sum((true_binary == 0) & (pred_binary == 0))

    # Compute Heidke Skill Score
    denominator = (a+b)*(b+d) + (a+c)*(c+d)
    return (a,b,c,d)
#     return round(2*(a*d - b*c) / denominator,2)

def compute_tss(truth, prediction):

    # https://stackoverflow.com/questions/56203875/how-to-compute-false-positive-rate-fpr-and-false-negative-rate-percantage
    tpr = recall_score(truth, prediction)  # it is better to name it y_test
    # to calculate, tnr we need to set the positive label to the other class
    # I assume your negative class consists of 0, if it is -1, change 0 below to that value
    tnr = recall_score(truth, prediction, pos_label=0)
    fpr = 1 - tnr
    try:
        neg, pos = confusion_matrix(truth, prediction)
        return round(tpr - fpr, 2)
    except ValueError:
        return None
    
def sigma_GMM(pred_mean, pred_var):
    #\sigma = [\sigma1, \sigma2, ...]
    sigma = np.sqrt(pred_var.cpu().numpy()).reshape(-1,1).reshape(pred_var.shape)
    # \bar{\sigma^2}
    bar_sigma_sq = (sigma**2).mean(0)
    ## \mu = [\mu1. \mu2, ...]
    mu = pred_mean.cpu().numpy().reshape(-1,1).reshape(pred_mean.shape)
    ## \bar(\mu^2)
    bar_mu_sq = (mu**2).mean(0)
    ## (\bar{\mu})^2
    sq_bar_mu = (mu.mean(0))**2
    ## the standard divation of a GMM f = \sum 1\{n} f_{i} with f_{i} being Gaussian.
    sigma_f = np.sqrt(bar_sigma_sq + bar_mu_sq - sq_bar_mu)
    return sigma_f

# def test_PULKKINEN(file_path, OMNI_data, start_date, end_date, QoIs):
#     station_data = pd.read_pickle(file_path)
#     test_SuperMAG_raw = station_data.loc[start_date:end_date]
#     ## remove missing OMNI data
#     test_existing_OMNI = OMNI_data.index.intersection(test_SuperMAG_raw.index)
#     ## now find the available OMNI data
#     test_OMNI_X = OMNI_data.loc[test_existing_OMNI]
#     ## find superMAG data that match OMNI data
#     test_SuperMAG = test_SuperMAG_raw.loc[test_existing_OMNI]
#     test_Y_raw = test_SuperMAG[QoIs]
#     test_X_raw = pd.concat([test_OMNI_X, test_SuperMAG[["SM_x", "SM_y", "SM_z"]]],
#                            axis = 1)
#     return test_X_raw, test_Y_raw
# def test_general(file_path, OMNI_data, QoIs,
#                  storm_date = datetime.datetime(2015, 3, 17, 4, 7),
#                  lower = 72,
#                  upper = 72):
#     station_data = pd.read_pickle(file_path)
#     start_date = storm_date - datetime.timedelta(hours=lower)
#     end_date = storm_date + datetime.timedelta(hours=upper)
#     test_SuperMAG_raw = station_data.loc[start_date:end_date]
#     ## remove missing OMNI data
#     test_existing_OMNI = OMNI_data.index.intersection(test_SuperMAG_raw.index)
#     ## now find the available OMNI data
#     test_OMNI_X = OMNI_data.loc[test_existing_OMNI]
#     ## find superMAG data that match OMNI data
#     test_SuperMAG = test_SuperMAG_raw.loc[test_existing_OMNI]
#     test_Y_raw = test_SuperMAG[QoIs]
#     test_X_raw = pd.concat([test_OMNI_X, test_SuperMAG[["SM_x", "SM_y", "SM_z"]]],
#                            axis = 1)
#     return test_X_raw, test_Y_raw

def test_PULKKINEN(file_path, OMNI_data, start_date, end_date, QoIs):
    station_data = pd.read_pickle(file_path)
    station_data = station_data.dropna()
    station_data.index = station_data['Time']
    station_data = station_data[[QoIs, "SM_lon", "SM_lat"]]
    test_SuperMAG_raw = station_data.loc[start_date:end_date]
    ## remove missing OMNI data
    test_existing_OMNI = OMNI_data.index.intersection(test_SuperMAG_raw.index)
    ## now find the available OMNI data
    test_OMNI_X = OMNI_data.loc[test_existing_OMNI]
    ## find superMAG data that match OMNI data
    test_SuperMAG = test_SuperMAG_raw.loc[test_existing_OMNI]
    test_Y_raw = test_SuperMAG[QoIs]
    test_X_raw = pd.concat([test_OMNI_X,
                            test_SuperMAG[["SM_lat"]],
                            np.cos(test_SuperMAG["SM_lon"].astype(np.float32) / 180 * np.pi),
                            np.sin(test_SuperMAG["SM_lon"].astype(np.float32) / 180 * np.pi)
                           ],
                           axis = 1)
    test_X_raw.columns = test_X_raw.columns[:-2].to_list() + ["CosLon", "SinLon"]
    return test_X_raw, test_Y_raw
def test_general(file_path, OMNI_data, QoIs,
                 storm_date = datetime.datetime(2015, 3, 17, 4, 7),
                 lower = 72,
                 upper = 72):
    station_data = pd.read_pickle(file_path)
    station_data = station_data.dropna()
    station_data.index = station_data['Time']
    station_data = station_data[[QoIs, "SM_lon", "SM_lat"]]
    start_date = storm_date - datetime.timedelta(hours=lower)
    end_date = storm_date + datetime.timedelta(hours=upper)
    test_SuperMAG_raw = station_data.loc[start_date:end_date]
    ## remove missing OMNI data
    test_existing_OMNI = OMNI_data.index.intersection(test_SuperMAG_raw.index)
    ## now find the available OMNI data
    test_OMNI_X = OMNI_data.loc[test_existing_OMNI]
    ## find superMAG data that match OMNI data
    test_SuperMAG = test_SuperMAG_raw.loc[test_existing_OMNI]
    test_Y_raw = test_SuperMAG[QoIs]
    test_X_raw = pd.concat([test_OMNI_X,
                            test_SuperMAG[["SM_lat"]],
                            np.cos(test_SuperMAG["SM_lon"].astype(np.float32) / 180 * np.pi),
                            np.sin(test_SuperMAG["SM_lon"].astype(np.float32) / 180 * np.pi)
                           ],
                           axis = 1)
    test_X_raw.columns = test_X_raw.columns[:-2].to_list() + ["CosLon", "SinLon"]
    return test_X_raw, test_Y_raw

# def compute_accuracy(model, data_loader, Image_folder, i, plot=True):
#     with torch.no_grad():
#         pred_means, pred_var, lls = model.predict(data_loader)
        
#         test_Y_OG = scalerY.inverse_transform(data_loader.dataset.Y.numpy().reshape(-1,1))
#         pred_Y_OG = scalerY.inverse_transform(pred_means.mean(0).cpu().numpy().reshape(-1,1))
#         if plot:
#             plt.plot(test_Y_OG, color = "black")
#             plt.plot(pred_Y_OG, color="red")
#             plt.savefig(Image_folder+ 'Epoch_%03d.png' % i, facecolor = 'white')
#             plt.close()
#         rmse = np.sqrt(np.mean((test_Y_OG - pred_Y_OG)**2))
#         return rmse
# def compute_accuracy(model, data_loader, Image_folder, i, boxcox_lmbda, plot=True):
#     with torch.no_grad():
#         pred_means, pred_var, lls = model.predict(data_loader)
        
#         test_Y_OG = scalerY.inverse_transform(data_loader.dataset.Y.numpy().reshape(-1,1))
#         pred_Y_OG = scalerY.inverse_transform(pred_means.mean(0).cpu().numpy().reshape(-1,1))
        
#         test_Y_OG = scipy.special.inv_boxcox(test_Y_OG, boxcox_lmbda, out=None)
#         pred_Y_OG = scipy.special.inv_boxcox(pred_Y_OG, boxcox_lmbda, out=None)
#         if plot:
#             plt.plot(test_Y_OG, color = "black")
#             plt.plot(pred_Y_OG, color="red")
#             plt.savefig(Image_folder+ 'Epoch_%03d.png' % i, facecolor = 'white')
#             plt.close()
#         rmse = np.sqrt(np.nanmean((test_Y_OG - pred_Y_OG)**2))
#         return rmse

def invert_yeojhonson(value, lmbda):
    if value>= 0 and lmbda == 0:
        return exp(value) - 1
    elif value >= 0 and lmbda != 0:
        return (value * lmbda + 1) ** (1 / lmbda) - 1
    elif value < 0 and lmbda != 2:
        return 1 - (-(2 - lmbda) * value + 1) ** (1 / (2 - lmbda))
    elif value < 0 and lmbda == 2:
        return 1 - exp(-value)
    
# Function to convert year, day of year, hour, and minute into a datetime object
def convert_to_datetime(row):
    return datetime.datetime(year=int(row[0]), month=int(row[1]), day=int(row[2]),
                             hour=int(row[3]), minute=int(row[4]))
def test_general_matchDAGGER(file_path, OMNI_data, QoIs, storm_start, storm_end):
    station_data = pd.read_pickle(file_path)
    station_data = station_data.dropna()
    station_data.index = station_data['Time']
    station_data = station_data[[QoIs, "SM_lon", "SM_lat"]]
    test_SuperMAG_raw = station_data.loc[storm_start:storm_end]
    ## remove missing OMNI data
    test_existing_OMNI = OMNI_data.index.intersection(test_SuperMAG_raw.index)
    ## now find the available OMNI data
    test_OMNI_X = OMNI_data.loc[test_existing_OMNI]
    ## find superMAG data that match OMNI data
    test_SuperMAG = test_SuperMAG_raw.loc[test_existing_OMNI]
    test_Y_raw = test_SuperMAG[QoIs]
    test_X_raw = pd.concat([test_OMNI_X,
                            test_SuperMAG[["SM_lat"]],
                            np.cos(test_SuperMAG["SM_lon"].astype(np.float32) / 180 * np.pi),
                            np.sin(test_SuperMAG["SM_lon"].astype(np.float32) / 180 * np.pi)
                           ],
                           axis = 1)
    test_X_raw.columns = test_X_raw.columns[:-2].to_list() + ["CosLon", "SinLon"]
    return test_X_raw, test_Y_raw
def test_general_matchGeospace(file_path, OMNI_data, QoIs, index):
    station_data = pd.read_pickle(file_path)
    station_data = station_data.dropna()
    station_data.index = station_data['Time']
    station_data = station_data[[QoIs, "SM_lon", "SM_lat"]]
    start_date = index[0].to_pydatetime()
    end_date = index[-1].to_pydatetime()
    test_SuperMAG_raw = station_data.loc[start_date:end_date]
    ## remove missing OMNI data
    test_existing_OMNI = OMNI_data.index.intersection(test_SuperMAG_raw.index)
    ## now find the available OMNI data
    test_OMNI_X = OMNI_data.loc[test_existing_OMNI]
    ## find superMAG data that match OMNI data
    test_SuperMAG = test_SuperMAG_raw.loc[test_existing_OMNI]
    test_Y_raw = test_SuperMAG[QoIs]
    test_X_raw = pd.concat([test_OMNI_X,
                            test_SuperMAG[["SM_lat"]],
                            np.cos(test_SuperMAG["SM_lon"].astype(np.float32) / 180 * np.pi),
                            np.sin(test_SuperMAG["SM_lon"].astype(np.float32) / 180 * np.pi)
                           ],
                           axis = 1)
    test_X_raw.columns = test_X_raw.columns[:-2].to_list() + ["CosLon", "SinLon"]
    return test_X_raw, test_Y_raw


# In[ ]:


class SuperMAGDataset(Dataset):
    def __init__(self, X_path, Y_path):
        self.X = torch.load(X_path)
        self.Y = torch.load(Y_path)
    
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        return {'X': self.X[idx], "Y": self.Y[idx]}


# In[ ]:


class Matern12(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=256, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(Matern12, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(MaternKernel(0.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))


# In[ ]:


class Matern32(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(Matern32, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(MaternKernel(1.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))


# In[ ]:


class SpecetralDeltaLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(SpecetralDeltaLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(SpectralDeltaKernel(
            num_dims=input_dims,
            num_deltas=1000
        ))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))


# In[ ]:


num_hidden1_dims = 20
num_hidden2_dims = 10
num_hidden3_dims = 10
class DeepGaussianProcess(DeepGP):
    def __init__(self, input_shape):
        hidden_layer_1 = Matern12(
            input_dims=input_shape,
            output_dims=num_hidden1_dims,
            mean_type='linear',
        )
        hidden_layer_2 = Matern32(
            input_dims=hidden_layer_1.output_dims,
            output_dims=num_hidden2_dims,
            mean_type='linear',
        )
        hidden_layer_3 = Matern32(
            input_dims=hidden_layer_2.output_dims,
            output_dims=num_hidden3_dims,
            mean_type='linear',
        )
        last_layer = SpecetralDeltaLayer(
            input_dims=hidden_layer_3.output_dims,
            output_dims=None,
            mean_type='linear',
        )
        super().__init__()

        self.hidden_layer1 = hidden_layer_1
        self.hidden_layer2 = hidden_layer_2
        self.hidden_layer3 = hidden_layer_3
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer1(inputs)
        hidden_rep2 = self.hidden_layer2(hidden_rep1)
        hidden_rep3 = self.hidden_layer3(hidden_rep2)
        output = self.last_layer(hidden_rep3)
        return output

    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for batch in test_loader:
                x_batch = batch['X'].to(DEVICE)
                y_batch = batch['Y'].to(DEVICE)
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(model.likelihood.log_marginal(y_batch, model(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)


# In[ ]:


class SuperMAGDataset_test(Dataset):
    def __init__(self, storm_X_tensor, storm_Y_tensor):
        self.X = storm_X_tensor
        self.Y = storm_Y_tensor
    
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        return {'X': self.X[idx], "Y": self.Y[idx]}


# In[ ]:


## 5 minutes window median.
path_prefix = "J:"
path = path_prefix+"/Paper_deltaB/data/train/"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
yeojohnson_lmbda = 1
window_max = 10
make_TestSet2_figure = True
make_TestSet2 = False
hour = 1
# station_path = path + 'data/QoIs_SM_20min_SM_lon_lat/'
# N_WORKERS = 10
# assert N_WORKERS<= os.cpu_count()
# QoIs = "dBN"
# year = 2011
station_to_visualize = sorted(['ABK', 'NEW', 'OTT', 'MEA', 'WNG', 'YKC'])
torch.manual_seed(20241029)


# In[ ]:


def DGP_pred(storm_X, storm_Y, model_in):
    if not storm_X.shape[0] == 0:
        ## standardize data
        storm_Y = scipy.stats.yeojohnson(storm_Y.to_numpy().reshape(-1,1), lmbda=yeojohnson_lmbda)
        storm_X = scalerX.transform(storm_X.to_numpy())
        storm_Y = scalerY.transform(storm_Y)
        storm_X = torch.from_numpy(storm_X).type(torch.float32).to(get_device())
        storm_Y = torch.from_numpy(storm_Y).type(torch.float32).to(get_device())
        storm_Y = storm_Y.reshape(storm_Y.shape[0])
        storm_dataset = SuperMAGDataset_test(storm_X, storm_Y)
        storm_loader = DataLoader(storm_dataset, batch_size=4096)

        with torch.no_grad():
            pred_means, pred_var, lls = model_in.predict(storm_loader)
            pred_Y_DGP = pred_means.mean(0).cpu().numpy().reshape(-1,1)
            storm_Y_OG = scalerY.inverse_transform(storm_loader.dataset.Y.cpu().numpy().reshape(-1,1))
            pred_Y_OG = scalerY.inverse_transform(pred_Y_DGP)
            storm_Y_OG = np.array([invert_yeojhonson(x, yeojohnson_lmbda) for x in storm_Y_OG])
            pred_Y_OG = np.array([invert_yeojhonson(x, yeojohnson_lmbda) for x in pred_Y_OG])
            ## prediction interval
            sigma = sigma_GMM(pred_means, pred_var)
            qnorm_975 = scipy.stats.norm.ppf(0.975, loc=0, scale=1)
            pred_Y_DGP_lr = pred_Y_DGP - qnorm_975*sigma.reshape(-1,1)
            pred_Y_DGP_up = pred_Y_DGP + qnorm_975*sigma.reshape(-1,1)
            pred_Y_OG_lr = np.array([invert_yeojhonson(x, yeojohnson_lmbda)
                                     for x in scalerY.inverse_transform(pred_Y_DGP_lr)]).flatten()
            pred_Y_OG_up = np.array([invert_yeojhonson(x, yeojohnson_lmbda)
                                     for x in scalerY.inverse_transform(pred_Y_DGP_up)]).flatten()
    return storm_Y_OG, pred_Y_OG, pred_Y_OG_lr, pred_Y_OG_up


# In[ ]:


if make_TestSet2:
    for year in [2011, 2015]:
        for QoIs in ['dBN', 'dBE', 'dBH']:
            with open(path + "scaler/scalerX_%s.pkl" % window_max,'rb') as f:
                scalerX = pickle.load(f)
            with open(path + "scaler/scalerY_%s_%s.pkl" % (QoIs, window_max),'rb') as f:
                scalerY = pickle.load(f)
            X_shape = 96
            ## set results path
            res_path = path_prefix+"/Paper_deltaB/data/test_station/TestSet2/%sT/%s/" % (window_max, QoIs)
            os.makedirs(res_path, exist_ok = True)
#             ## load model
#             OMNI = pd.read_pickle(path + "OMNI_1995_2022_5m_feature.pkl").dropna()
#             model = DeepGaussianProcess(X_shape)
#             state_dict = torch.load(path + "model/paper/model_state_%s_10T.pth" % QoIs)
#             model.load_state_dict(state_dict)
            ## load \delta_SW + 1h model
            OMNI = pd.read_pickle(path + "../Input/OMNI_paper_5m_feature.pkl").dropna()
            deltaT = datetime.timedelta(hours=hour)
            deltaT_SW = datetime.timedelta(minutes=45)
            OMNI.index = OMNI.index + deltaT
            ## 1-hour ahead
            model = DeepGaussianProcess(X_shape)
            state_dict_ahead = torch.load(path + "model/paper/model_state_%s_10T_%sh_ahead.pth" % (QoIs, hour))
            model.load_state_dict(state_dict_ahead)
            ## model persistence
            model_p = DeepGaussianProcess(X_shape)
            state_dict = torch.load(path + "model/paper/model_state_%s_10T.pth" % QoIs)
            model_p.load_state_dict(state_dict)
            if torch.cuda.is_available():
                model = model.cuda()
                model_p = model_p.cuda()
            
            ## load station filenames
            # QoIs_SM_20min_SM_lon_lat_1995_2002
            # station_path = path + '../QoIs_SM_20min_SM_lon_lat_1995_2002/'
            station_path = path_prefix+"/Paper_deltaB/data/AllStations_AllYear_1min_raw/"
            station_file = sorted([x for x in os.listdir(station_path) if bool(re.match(r'.*_%s\.pkl$' % year, x))])
            dagger_time = np.load(path_prefix+'/Paper_deltaB/data/DAGGER/%s/dagger_time_%s.npy' % (year,year))
            dagger_stations = np.load(path_prefix+'/Paper_deltaB/data/DAGGER/%s/dagger_stations_%s.npy' % (year,year))
            dagger_pred = np.load(path_prefix+'/Paper_deltaB/data/DAGGER/%s/dagger_%s_%s.npy' % (year, QoIs, year))
#             Geospace_path = "J:/deltaB/data/Geospace/"
#             Geospace_stations= sorted([x.split('.csv')[0] for x in os.listdir(Geospace_path)])
            My_stations = sorted([x.split('_%s' % year)[0] for x in station_file])
            station_to_evaluate = sorted(list(set(My_stations) & set(dagger_stations.tolist())))
            station_to_evaluate_file = sorted([x for x in station_file if (x.split('_%s' % year)[0] in station_to_evaluate)])
            if year == 2015:
                test_storm_date_list = [datetime.datetime(2015, 3, 17, 4, 7)]
            else:
                test_storm_date_list = [datetime.datetime(2011, 8, 5, 18, 2)]
            threholds = [50, 100, 200, 300, 400]
            for j in range(len(test_storm_date_list)):
                storm_onset = test_storm_date_list[j]
                storm_start = dagger_time[0]
                storm_end = dagger_time[-1]
                storm_folder_path = res_path + storm_onset.strftime("%Y_%m_%d") +'tripleComparison' 
                os.makedirs(storm_folder_path, exist_ok=True)
                HSS_df = []
                for i in range(len(station_to_evaluate_file)):
#                     # Geospace model predictions: --------------------------------------------------------------------------
#                     GeospaceSim_file = Geospace_path + station_to_evaluate_file[i].split('_%s.pkl' % year)[0] + '.csv'
#                     df = pd.read_csv(GeospaceSim_file)
#                     ## Apply the function to each row in the DataFrame
#                     df['Datetime'] = df.apply(convert_to_datetime, axis=1)
#                     df.set_index('Datetime', inplace=True)
#                     if QoIs == "dBH":
#                         df['%sSim' % QoIs] = np.sqrt(df['BeSim']**2 + df['BnSim']**2)
#                         df['%sObs' % QoIs] = np.sqrt(df['BeObs']**2 + df['BnObs']**2)
#                     elif QoIs == "dBE":
#                         df['%sSim' % QoIs] = df['BeSim']
#                         df['%sObs' % QoIs] = df['BeObs']
#                     else:
#                         df['%sSim' % QoIs] = df['BnSim']
#                         df['%sObs' % QoIs] = df['BnObs']
#                     storm_Geospace = df[storm_start:storm_end].drop_duplicates()
                    # Deep GP model predictions: --------------------------------------------------------------------------
                    file_path = station_path + station_to_evaluate_file[i]
                    station_data = pd.read_pickle(file_path)
                    station_data = station_data.dropna()
                    station_data.index = station_data['Time']
                    station_data = station_data[[QoIs, "SM_lon", "SM_lat"]]
#                   storm_X, storm_Y = test_general_matchGeospace(file_path, OMNI, QoIs,storm_Geospace.index)
                    storm_X, storm_Y = test_general_matchDAGGER(file_path, OMNI, QoIs,
                                                                              storm_start, storm_end)
#                         Joint_index = storm_Y.index.intersection(storm_Geospace.index)
#                         Joint_index = storm_Y_ahead.index.intersection(Joint_index)
                    dagger_station_idx = np.where(dagger_stations == station_to_evaluate_file[i].split('_%s.pkl' % year)[0])[0].item()
                    dagger_pd = pd.DataFrame(dagger_pred[:,dagger_station_idx])
                    dagger_pd.index = dagger_time
                    dagger_pd.columns = ['%sDagger'% QoIs]
                    Joint_index = storm_Y.index.intersection(dagger_pd.index)
                    Joint_index = station_data.index.intersection(Joint_index-deltaT-deltaT_SW)+deltaT+deltaT_SW
#                     Joint_index = dagger_pd.index.intersection(Joint_index)
#                         storm_X = storm_X.loc[Joint_index]
#                         storm_Y = storm_Y.loc[Joint_index]
                    storm_X = storm_X.loc[Joint_index]
                    storm_Y = storm_Y.loc[Joint_index]
                    if not storm_X.shape[0] == 0:
                        ## 1-h prediction
                        storm_Y_OG, pred_Y_OG, pred_Y_OG_lr, pred_Y_OG_up = DGP_pred(storm_X,storm_Y,model)
                        ## model persistence
                        storm_Y_OG_p, pred_Y_OG_p, pred_Y_OG_lr_p, pred_Y_OG_up_p = DGP_pred(storm_X,storm_Y,model_p)
                        ## the observation persistence
                        observation_persistence = station_data.loc[Joint_index-deltaT-deltaT_SW][QoIs].to_numpy().reshape(-1,1)
                        # HSS calculation: -----------------------------------------------------------------------
                        ## Deep GP HSS calculation
#                             HSS = []
#                             for threshold in threholds:
#                                 HSS.append(np.array(compute_hss_count(np.sign(storm_Y_OG) * storm_Y_OG,
#                                               np.sign(storm_Y_OG) * pred_Y_OG,
#                                               threshold)))
#                             HSS_pd = pd.DataFrame(np.concatenate(HSS)).T
#                             HSS_pd["Station"] = station_to_evaluate_file[i][0:3]
#                             MAE = mean_absolute_error(storm_Y_OG, pred_Y_OG)
#                             HSS_pd["MAE"] = int(MAE)
#                             HSS_pd["N"] = pred_Y_OG.shape[0]
#                             HSS_pd["SignRateCorrectNum"] = np.sum(np.sign(storm_Y_OG) == np.sign(pred_Y_OG))
#                             HSS_pd["Method"] = "GeoDGP"
#                             HSS_df.append(HSS_pd)
                        ## Deep GP 1h ahead HSS calculation
                        HSS = []
                        for threshold in threholds:
                            HSS.append(np.array(compute_hss_count(np.sign(storm_Y_OG) * storm_Y_OG,
                                          np.sign(storm_Y_OG) * pred_Y_OG,
                                          threshold)))
                        HSS_pd = pd.DataFrame(np.concatenate(HSS)).T
                        HSS_pd["Station"] = station_to_evaluate_file[i][0:3]
                        MAE = mean_absolute_error(storm_Y_OG, pred_Y_OG)
                        HSS_pd["MAE"] = int(MAE)
                        HSS_pd["N"] = pred_Y_OG.shape[0]
                        HSS_pd["SignRateCorrectNum"] = np.sum(np.sign(storm_Y_OG) == np.sign(pred_Y_OG))
                        HSS_pd["Method"] = "GeoDGP_ahead"
                        HSS_df.append(HSS_pd)
                        ## model persistence HSS calculation
                        HSS = []
                        for threshold in threholds:
                            HSS.append(np.array(compute_hss_count(np.sign(storm_Y_OG) * storm_Y_OG,
                                          np.sign(storm_Y_OG) * pred_Y_OG_p,
                                          threshold)))
                        HSS_pd = pd.DataFrame(np.concatenate(HSS)).T
                        HSS_pd["Station"] = station_to_evaluate_file[i][0:3]
                        MAE = mean_absolute_error(storm_Y_OG, pred_Y_OG_p)
                        HSS_pd["MAE"] = int(MAE)
                        HSS_pd["N"] = pred_Y_OG_p.shape[0]
                        HSS_pd["SignRateCorrectNum"] = np.sum(np.sign(storm_Y_OG) == np.sign(pred_Y_OG_p))
                        HSS_pd["Method"] = "GeoDGP_persistence"
                        HSS_df.append(HSS_pd)
#                             ## Geospace HSS calculation
#                             HSS = []
#                             Geospace_Y_OG = storm_Geospace.loc[Joint_index]['%sSim' % QoIs].to_numpy().reshape(-1,1)
#                             for threshold in threholds:
#                                 HSS.append(np.array(compute_hss_count(np.sign(storm_Y_OG) * storm_Y_OG,
#                                                                       np.sign(storm_Y_OG) * Geospace_Y_OG,
#                                                                       threshold)))
#                             HSS_pd = pd.DataFrame(np.concatenate(HSS)).T
#                             HSS_pd["Station"] = station_to_evaluate_file[i][0:3]
#                             MAE = mean_absolute_error(storm_Y_OG, Geospace_Y_OG)
#                             HSS_pd["MAE"] = int(MAE)
#                             HSS_pd["N"] = pred_Y_OG.shape[0]
#                             HSS_pd["SignRateCorrectNum"] = np.sum(np.sign(storm_Y_OG) ==  np.sign(Geospace_Y_OG))
#                             HSS_pd["Method"] = "Geospace"
#                             HSS_df.append(HSS_pd)
                        ## Dagger HSS calculation
                        HSS = []
                        Dagger_Y_OG = dagger_pd.loc[Joint_index]['%sDagger' % QoIs].to_numpy().reshape(-1,1)
                        for threshold in threholds:
                            HSS.append(np.array(compute_hss_count(np.sign(storm_Y_OG) * storm_Y_OG,
                                                                  np.sign(storm_Y_OG) * Dagger_Y_OG,
                                                                  threshold)))
                        HSS_pd = pd.DataFrame(np.concatenate(HSS)).T
                        HSS_pd["Station"] = station_to_evaluate_file[i][0:3]
                        MAE = mean_absolute_error(storm_Y_OG, Dagger_Y_OG)
                        HSS_pd["MAE"] = int(MAE)
                        HSS_pd["N"] = pred_Y_OG.shape[0]
                        HSS_pd["SignRateCorrectNum"] = np.sum(np.sign(storm_Y_OG) ==  np.sign(Dagger_Y_OG))
                        HSS_pd["Method"] = "Dagger"
                        HSS_df.append(HSS_pd)
                        ## Observation Persistence HSS calculation
                        HSS = []
                        for threshold in threholds:
                            HSS.append(np.array(compute_hss_count(np.sign(storm_Y_OG) * storm_Y_OG,
                                                                  np.sign(storm_Y_OG) * observation_persistence,
                                                                  threshold)))
                        HSS_pd = pd.DataFrame(np.concatenate(HSS)).T
                        HSS_pd["Station"] = station_to_evaluate_file[i][0:3]
                        MAE = mean_absolute_error(storm_Y_OG, observation_persistence)
                        HSS_pd["MAE"] = int(MAE)
                        HSS_pd["N"] = pred_Y_OG.shape[0]
                        HSS_pd["SignRateCorrectNum"] = np.sum(np.sign(storm_Y_OG) ==  np.sign(observation_persistence))
                        HSS_pd["Method"] = "Observation_persistence"
                        HSS_df.append(HSS_pd)
                if len(HSS_df) >0:
                    HSS_df = pd.concat(HSS_df, axis=0)
                    first_column = HSS_df.pop('Station') 

                    # insert column using insert(position,column_name, 
                    # first_column) function 
                    HSS_df.insert(0, 'Station', first_column) 
                    HSS_df.to_csv(res_path+"ModelComp_HSS_%s.csv" % test_storm_date_list[j].strftime("%Y_%m_%d"), index=False)
                    print('Year: %s, QoIs: %s, finished' % (year, QoIs))
                else:
                    print("Storm %s data not available" % test_storm_date_list[j].strftime("%Y_%m_%d"))


# In[ ]:


if make_TestSet2_figure:
    for year in [2011]:
        for QoIs in ['dBH']:
            with open(path + "scaler/scalerX_%s.pkl" % window_max,'rb') as f:
                scalerX = pickle.load(f)
            with open(path + "scaler/scalerY_%s_%s.pkl" % (QoIs, window_max),'rb') as f:
                scalerY = pickle.load(f)
            X_shape = 96
            ## set results path
            res_path = path_prefix+"/Paper_deltaB/data/test_station/TestSet2/%sT/%s/" % (window_max, QoIs)
            os.makedirs(res_path, exist_ok = True)
#             ## load model
#             OMNI = pd.read_pickle(path + "OMNI_1995_2022_5m_feature.pkl").dropna()
#             model = DeepGaussianProcess(X_shape)
#             state_dict = torch.load(path + "model/paper/model_state_%s_10T.pth" % QoIs)
#             model.load_state_dict(state_dict)
            ## load \delta_SW + 1h model
            OMNI = pd.read_pickle(path + "../Input/OMNI_paper_5m_feature.pkl").dropna()
            deltaT = datetime.timedelta(hours=hour)
            OMNI.index = OMNI.index + deltaT
            model = DeepGaussianProcess(X_shape)
            state_dict_ahead = torch.load(path + "model/paper/model_state_%s_10T_%sh_ahead.pth" % (QoIs, hour))
            model.load_state_dict(state_dict_ahead)
            if torch.cuda.is_available():
#                 model = model.cuda()
                model = model.cuda()
            
            ## load station filenames
            # QoIs_SM_20min_SM_lon_lat_1995_2002
            # station_path = path + '../QoIs_SM_20min_SM_lon_lat_1995_2002/'
            station_path = path_prefix+"/Paper_deltaB/data/AllStations_AllYear_1min_raw/"
            station_file = sorted([x for x in os.listdir(station_path) if bool(re.match(r'.*_%s\.pkl$' % year, x))])
            dagger_time = np.load(path_prefix+'/Paper_deltaB/data/DAGGER/%s/dagger_time_%s.npy' % (year,year))
            dagger_stations = np.load(path_prefix+'/Paper_deltaB/data/DAGGER/%s/dagger_stations_%s.npy' % (year,year))
            dagger_pred = np.load(path_prefix+'/Paper_deltaB/data/DAGGER/%s/dagger_%s_%s.npy' % (year, QoIs, year))
#             Geospace_path = "J:/deltaB/data/Geospace/"
#             Geospace_stations= sorted([x.split('.csv')[0] for x in os.listdir(Geospace_path)])
            My_stations = sorted([x.split('_%s' % year)[0] for x in station_file])
            station_to_evaluate = sorted(list(set(My_stations) & set(dagger_stations.tolist())))
            station_to_evaluate_file = sorted([x for x in station_file if (x.split('_%s' % year)[0] in station_to_evaluate)])
            if year == 2015:
                test_storm_date_list = [datetime.datetime(2015, 3, 17, 4, 7)]
            else:
                test_storm_date_list = [datetime.datetime(2011, 8, 5, 18, 2)]
            threholds = [50, 100, 200, 300, 400]
            plt.rcParams.update({'font.size': 17})
            fig, axes = plt.subplots(3, 2, figsize=(24, 13))
            for j in range(len(test_storm_date_list)):
                storm_onset = test_storm_date_list[j]
                storm_start = dagger_time[0]
                storm_end = dagger_time[-1]
                storm_folder_path = res_path + storm_onset.strftime("%Y_%m_%d") +'tripleComparison' 
                os.makedirs(storm_folder_path, exist_ok=True)
                HSS_df = []
                for i in range(len(station_to_evaluate_file)):
                    if station_to_evaluate_file[i][0:3] in station_to_visualize:
    #                     # Geospace model predictions: --------------------------------------------------------------------------
    #                     GeospaceSim_file = Geospace_path + station_to_evaluate_file[i].split('_%s.pkl' % year)[0] + '.csv'
    #                     df = pd.read_csv(GeospaceSim_file)
    #                     ## Apply the function to each row in the DataFrame
    #                     df['Datetime'] = df.apply(convert_to_datetime, axis=1)
    #                     df.set_index('Datetime', inplace=True)
    #                     if QoIs == "dBH":
    #                         df['%sSim' % QoIs] = np.sqrt(df['BeSim']**2 + df['BnSim']**2)
    #                         df['%sObs' % QoIs] = np.sqrt(df['BeObs']**2 + df['BnObs']**2)
    #                     elif QoIs == "dBE":
    #                         df['%sSim' % QoIs] = df['BeSim']
    #                         df['%sObs' % QoIs] = df['BeObs']
    #                     else:
    #                         df['%sSim' % QoIs] = df['BnSim']
    #                         df['%sObs' % QoIs] = df['BnObs']
    #                     storm_Geospace = df[storm_start:storm_end].drop_duplicates()
                        # Deep GP model predictions: --------------------------------------------------------------------------
                        file_path = station_path + station_to_evaluate_file[i]
    #                         storm_X, storm_Y = test_general_matchGeospace(file_path, OMNI, QoIs,storm_Geospace.index)
                        storm_X, storm_Y = test_general_matchDAGGER(file_path, OMNI, QoIs,
                                                                                  storm_start, storm_end)
    #                         Joint_index = storm_Y.index.intersection(storm_Geospace.index)
    #                         Joint_index = storm_Y_ahead.index.intersection(Joint_index)
                        dagger_station_idx = np.where(dagger_stations == 
                                                      station_to_evaluate_file[i].split('_%s.pkl' % year)[0])[0].item()
                        dagger_pd = pd.DataFrame(dagger_pred[:,dagger_station_idx])
                        dagger_pd.index = dagger_time
                        dagger_pd.columns = ['%sDagger'% QoIs]
                        Joint_index = storm_Y.index.intersection(dagger_pd.index)
    #                     Joint_index = dagger_pd.index.intersection(Joint_index)
    #                         storm_X = storm_X.loc[Joint_index]
    #                         storm_Y = storm_Y.loc[Joint_index]
                        storm_X = storm_X.loc[Joint_index]
                        storm_Y = storm_Y.loc[Joint_index]
                        if not storm_X.shape[0] == 0:
    #                             storm_Y_OG, pred_Y_OG = DGP_pred(storm_X, storm_Y, model)
                            storm_Y_OG, pred_Y_OG, pred_Y_OG_lr, pred_Y_OG_up = DGP_pred(storm_X, storm_Y, model)
                            # visualization: -----------------------------------------------------------------------

                            index = np.where(np.array(station_to_visualize) == station_to_evaluate_file[i][0:3])[0][0]
                            ax = axes.flat[index]
                            time_idx = dagger_pd.loc[Joint_index]['%sDagger'%QoIs].resample('1T').median().index
                            time_idx = time_idx[time_idx < datetime.datetime(2011, 8,6, 12)]
                            ## Observation
                            storm_Y_OG_df = pd.DataFrame(storm_Y_OG)
                            storm_Y_OG_df.index = Joint_index
                            storm_Y_OG_df.columns = ['%s' % QoIs]
                            ax.plot(time_idx, storm_Y_OG_df['%s' % QoIs].resample('1T').median().loc[time_idx],
                                     color = "black", label = "Station", linewidth=2.3)
                            ## Dagger
                            ax.plot(time_idx, dagger_pd.loc[Joint_index]['%sDagger'%QoIs].resample('1T').median().loc[time_idx],
                                     color = "seagreen", label = "DAGGER " + r'$(T_{S}+0.5h)$', linewidth=1, alpha = 0.8)
                            ## Deep GP 1h + SW ahead
                            pred_Y_OG_df = pd.DataFrame(pred_Y_OG)
                            pred_Y_OG_df.index = Joint_index
                            pred_Y_OG_df.columns = ['%s' % QoIs]
                            ax.plot(time_idx, pred_Y_OG_df['%s' % QoIs].resample('1T').median().loc[time_idx],
                                     color= "red", label='GeoDGP ' + r'$(T_{S}+1h)$' , linewidth=1)
                            ax.tick_params(axis='x', rotation=10)
                            ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))

                            pred_Y_OG_lr = pd.DataFrame(pred_Y_OG_lr)
                            pred_Y_OG_lr.index = Joint_index
                            pred_Y_OG_lr.columns = ['%s' % QoIs]
                            pred_Y_OG_up = pd.DataFrame(pred_Y_OG_up)
                            pred_Y_OG_up.index = Joint_index
                            pred_Y_OG_up.columns = ['%s' % QoIs]
                            ax.fill_between(time_idx,
                                             np.clip(pred_Y_OG_lr['%s' % QoIs].resample('1T').median().loc[time_idx],
                                                     a_min=0, a_max=3000),
                                             pred_Y_OG_up['%s' % QoIs].resample('1T').median().loc[time_idx],
                                             color='lightgrey', alpha=.7)
                            ax.text(0.82, .95, station_to_evaluate_file[i][0:3],
                                    ha='left', va='top', transform=ax.transAxes,
                                    fontsize=30)
                            if index == 5:
                                params = {'legend.fontsize': 25, 'legend.handlelength': 4}
                                plt.rcParams.update(params)
                                handles, labels = plt.gca().get_legend_handles_labels()
                                leg = plt.figlegend(handles, labels, loc = 'lower center', ncol=3,
                                                    labelspacing=15, frameon=False)
                                for legobj in leg.legend_handles:
                                    legobj.set_linewidth(10)
                                fig.text(0.07, 0.5, '%s (nT)' % QoIs, va='center', rotation='vertical', fontsize=30)
                                plt.savefig(path_prefix+"/Paper_deltaB/figure/TestSet2/TestSet2_stations_%s_%s.png" % (QoIs, year),
                                            bbox_inches='tight', dpi=300)
                                plt.close()

