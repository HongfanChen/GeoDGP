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
import matplotlib.dates as mdates


# In[ ]:


def get_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def df_to_tensor(x: pd.DataFrame) -> torch:
    return torch.from_numpy(x.values).to(get_device())

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_hss(true_values, predictions, threshold):
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

    return round(2*(a*d - b*c) / denominator,2)

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
                            np.cos(test_SuperMAG["SM_lon"]/ 180 * np.pi),
                            np.sin(test_SuperMAG["SM_lon"] / 180 * np.pi)
                           ],
                           axis = 1)
    test_X_raw.columns = test_X_raw.columns[:-2].to_list() + ["CosLon", "SinLon"]
    return test_X_raw, test_Y_raw
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
#     test_X_raw = pd.concat([test_OMNI_X,
#                             test_SuperMAG[["SM_lat"]],
#                             np.cos(test_SuperMAG["SM_lon"]/ 180 * np.pi),
#                             np.sin(test_SuperMAG["SM_lon"] / 180 * np.pi)
#                            ],
#                            axis = 1)
#     test_X_raw.columns = test_X_raw.columns[:-2].to_list() + ["CosLon", "SinLon"]
#     return test_X_raw, test_Y_raw
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

def invert_yeojhonson(value, lmbda):
    if value>= 0 and lmbda == 0:
        return exp(value) - 1
    elif value >= 0 and lmbda != 0:
        return (value * lmbda + 1) ** (1 / lmbda) - 1
    elif value < 0 and lmbda != 2:
        return 1 - (-(2 - lmbda) * value + 1) ** (1 / (2 - lmbda))
    elif value < 0 and lmbda == 2:
        return 1 - exp(-value) 
def convert_to_datetime(row):
    return datetime.datetime(year=int(row[0]), month=int(row[1]), day=int(row[2]),
                             hour=int(row[3]), minute=int(row[4]))


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


path_prefix = "J:"
path = path_prefix+"/Paper_deltaB/data/train/"
QoIs = "dBH"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
yeojohnson_lmbda = 1
window_max = 10
plot_image=True
multihour_ahead = False
with open(path + "scaler/scalerX_%s.pkl" % window_max,'rb') as f:
    scalerX = pickle.load(f)
with open(path + "scaler/scalerY_%s_%s.pkl" % (QoIs, window_max),'rb') as f:
    scalerY = pickle.load(f)
X_shape = 96
## OMNI data
# OMNI = pd.read_pickle(path + "OMNI_May10_5m_feature.pkl").dropna()
## IMF ballistic WIND fixBz data
# OMNI = pd.read_pickle(path + "OMNI_May10_5m_feature_ballistic_WIND_fixBz.pkl").dropna()
## IMF ACE data
deltaT_SW = datetime.timedelta(minutes=45)
if multihour_ahead:
    OMNI = pd.read_pickle(path + "../Input/OMNI_May10_5m_feature_ACE.pkl").dropna()
    deltaT = datetime.timedelta(hours=1)
    OMNI.index = OMNI.index +deltaT
    res_path = path_prefix+"/Paper_deltaB/data/test_station/TestSet3_IMF_ACE/%sT_1h_ahead/%s/" % (window_max, QoIs)
else:
    OMNI = pd.read_pickle(path + "../Input/OMNI_May10_5m_feature_ACE.pkl").dropna()   
    res_path = path_prefix+"/Paper_deltaB/data/test_station/TestSet3_IMF_ACE/%sT/%s/" % (window_max, QoIs)
## IMF DSCOVR + ACE
# OMNI = pd.read_pickle(path + "OMNI_May10_5m_feature_ACE.pkl").dropna()
## set results path
# res_path = "J:/Paper_deltaB/data/test_station/TestSet3/%sT/%s/" % (window_max, QoIs)
# res_path = "J:/Paper_deltaB/data/test_station/TestSet3_IMF_WIND/%sT/%s/" % (window_max, QoIs)
# res_path = "J:/Paper_deltaB/data/test_station/TestSet3_IMF_ACE/%sT/%s/" % (window_max, QoIs)
# res_path = "J:/Paper_deltaB/data/test_station/TestSet3_IMF_DSCOVR_ACE/%sT/%s/" % (window_max, QoIs)
os.makedirs(res_path, exist_ok = True)
make_FFTfile = False
make_TestSet3 = True
make_FFT_mergedfile = False
make_TestSet3_figure = False
station_to_evaluate_FFT = ['ABK', 'OTT', 'FRD']
station_to_visualize = sorted(['ABK', 'NEW', 'OTT', 'MEA', 'WNG', 'YKC'])
torch.manual_seed(20241030)


# In[ ]:


# start = '2024-05-10 20:00'
# end = '2024-05-11'
# quantity = 'Vx Velocity, km/s, GSE'
# plt.plot(OMNI_2d_prior[start:end][quantity], color='red')
# plt.plot(OMNI[start:end][quantity], color='black')
# plt.plot(OMNI_WIND[start:end][quantity], color='green')


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
class DeepGP(DeepGP):
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


if multihour_ahead:
    # 1-hour model
    model = DeepGP(X_shape)
    state_dict = torch.load(path + "model/paper/model_state_%s_10T_1h_ahead.pth" % QoIs)
    model.load_state_dict(state_dict)
    # model persistence
    model_p = DeepGP(X_shape)
    state_dict = torch.load(path + "model/paper/model_state_%s_10T.pth" % QoIs)
    model_p.load_state_dict(state_dict)
    ## load to GPU
    if torch.cuda.is_available():
        model = model.cuda()
        model_p = model_p.cuda()
else:
    model = DeepGP(X_shape)
    state_dict = torch.load(path + "model/paper/model_state_%s_10T.pth" % QoIs)
    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model = model.cuda()


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


## ---------------------------------------------------------------------------------------------------------------
## ------------------------------------------ Without Dst history ------------------------------------------------
## ---------------------------------------------------------------------------------------------------------------
# ## load OMNI
# OMNI = pd.read_pickle(path + "data/OMNI_PhysicsInformedCols.pkl").dropna()
# ## it actually matters a lot to take the lag first and then calculate the median.
# lagged_dst = []
# covariate = 'Dst'
# Dst_history = False
# ## up to 12 hours of Dst.
# for lag in [i*60 for i in range(1,13)]:
#     lagged_dst.append(OMNI[covariate].shift(lag).rename(f'{covariate}_lag_{1*lag}'))
# lagged_data = []
# for lag in [i*5 for i in range(1,13)]:
#     for covariate in OMNI.columns[0:len(OMNI.columns)-3]:  # 60-minute history, do not lag dipole tilt angle.
#         lagged_data.append(OMNI[covariate].shift(lag).rename(f'{covariate}_lag_{1*lag}'))
# if Dst_history:
#     OMNI = pd.concat([OMNI] + lagged_dst, axis=1)
# OMNI = pd.concat([OMNI] + lagged_data, axis=1)
# OMNI = OMNI.resample('5T').median().dropna()
# OMNI.index = OMNI.index + datetime.timedelta(minutes=5)

## ---------------------------------------------------------------------------------------------------------------
## --------------------------------------------- With Dst history ------------------------------------------------
## ---------------------------------------------------------------------------------------------------------------


# In[ ]:


'''Six storms in PULKKINEN ET. AL. 2013'''
'''Remove these from training'''
## Storm 1 2003-10-29 06:00 - 2003-10-30 06:00
## Storm 2 2006-12-14 12:00 - 2006-12-16 00:00
## Storm 3 2001-08-31 00:00 - 2001-09-01 00:00
## Storm 4 2005-08-31 10:00 - 2005-09-01 12:00
## Storm 5 2010-04-05 00:00 - 2010-04-06 00:00
## Storm 6 2011-08-05 09:00 - 2011-08-06 09:00
'''Storms in Year 2014 for validation plus minus 24 hours'''
'''Use these as validation'''
## 2014-02-23 06:55
## 2014-02-27 16:56
## 2014-04-11 05:54
## 2014-05-07 23:44
## 2014-06-07 16:36
## 2014-08-27 03:35
## 2014-09-12 05:20
# PULKKINEN_storm_date_list = [(datetime.datetime(2003, 10, 29, 6, 0),datetime.datetime(2003, 10, 30, 6, 0)),
#                              (datetime.datetime(2006, 12, 14, 12, 0),datetime.datetime(2006, 12, 16, 0, 0)),
#                              (datetime.datetime(2001, 8, 31, 0, 0),datetime.datetime(2001, 9, 1, 0, 0)),
#                              (datetime.datetime(2005, 8, 31, 10, 0),datetime.datetime(2005, 9, 1, 12, 0)),
#                              (datetime.datetime(2010, 4, 5, 0, 0),datetime.datetime(2010, 4, 6, 0, 0)),
#                              (datetime.datetime(2011, 8, 5, 9, 0),datetime.datetime(2011, 8, 6, 9, 0))]
# validation_storm_date_list = [datetime.datetime(2014, 2, 23, 6, 55),
#                              datetime.datetime(2014, 2, 27, 16, 56),
#                              datetime.datetime(2014, 4, 11, 5, 54),
#                              datetime.datetime(2014, 5, 7, 23, 44),
#                              datetime.datetime(2014, 6, 7, 16, 36),
#                              datetime.datetime(2014, 8, 27, 3, 35),
#                              datetime.datetime(2014, 9, 12, 5, 20)]
# test_storm_date_list = [datetime.datetime(2015, 2, 16, 19, 24),
#                         datetime.datetime(2015, 3, 17, 4, 7),
#                        datetime.datetime(2015, 4, 9, 21, 52),
#                        datetime.datetime(2015, 4, 14, 12, 55),
#                        datetime.datetime(2015, 5, 12, 18, 5),
#                        datetime.datetime(2015, 5, 18, 10, 12),
#                        datetime.datetime(2015, 6, 7, 10, 30),
#                        datetime.datetime(2015, 6, 22, 5, 0),
#                        datetime.datetime(2015, 7, 4, 13, 6),
#                        datetime.datetime(2015, 7, 10, 22, 21),
#                        datetime.datetime(2015, 7, 23, 1, 51),
#                        datetime.datetime(2015, 8, 15, 8, 4),
#                        datetime.datetime(2015, 8, 26, 5, 45),
#                        datetime.datetime(2015, 9, 7, 13, 13),
#                        datetime.datetime(2015, 9, 8, 21, 45),
#                        datetime.datetime(2015, 9, 20, 5, 46),
#                        datetime.datetime(2015, 10, 4, 0, 30),
#                        datetime.datetime(2015, 10, 7, 1, 41),
#                        datetime.datetime(2015, 11, 3, 5, 31),
#                        datetime.datetime(2015, 11, 6, 18, 9),
#                        datetime.datetime(2015, 11, 30, 6, 9),
#                        datetime.datetime(2015, 12, 19, 16, 13)]
May10_storm = [datetime.datetime(2024, 5, 10, 18, 0)]
# test_storm_date_list = [datetime.datetime(2016, 3, 6, 6, 0),
#                         datetime.datetime(2016, 3, 14, 17, 6),
#                        datetime.datetime(2016, 4, 2, 11, 47),
#                        datetime.datetime(2016, 4, 7, 16, 40),
#                        datetime.datetime(2016, 4, 12, 19, 38),
#                        datetime.datetime(2016, 4, 16, 17, 40),
#                        datetime.datetime(2016, 5, 8, 1, 4),
#                        datetime.datetime(2016, 6, 5, 5, 46),
#                        datetime.datetime(2016, 8, 2, 5, 42),
#                        datetime.datetime(2016, 8, 23, 10, 59),
#                        datetime.datetime(2016, 10, 12, 21, 47),
#                        datetime.datetime(2016, 12, 21, 9, 17)]
upper = 48
lower = 6


# In[ ]:


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


# def test_IH_AS_matchGeospace(file_path, OMNI_data, QoIs, index):
#     station_data = pd.read_pickle(file_path)
#     station_data = station_data.dropna()
#     station_data.index = station_data['Time']
#     station_data = station_data[[QoIs, "SM_lon", "SM_lat"]]
#     start_date = index[0].to_pydatetime()
#     end_date = index[-1].to_pydatetime()
#     test_SuperMAG_raw = station_data.loc[start_date:end_date]
#     ## remove missing OMNI data
#     test_existing_OMNI = OMNI_data.index.intersection(test_SuperMAG_raw.index)
#     ## now find the available OMNI data
#     test_OMNI_X = OMNI_data.loc[test_existing_OMNI]
#     ## find superMAG data that match OMNI data
#     test_SuperMAG = test_SuperMAG_raw.loc[test_existing_OMNI]
#     test_Y_raw = test_SuperMAG[QoIs]
#     test_X_raw = pd.concat([test_OMNI_X,
#                             test_SuperMAG[["SM_lat"]],
#                             np.cos(test_SuperMAG["SM_lon"].astype(np.float32) / 180 * np.pi),
#                             np.sin(test_SuperMAG["SM_lon"].astype(np.float32) / 180 * np.pi)
#                            ],
#                            axis = 1)
#     test_X_raw.columns = test_X_raw.columns[:-2].to_list() + ["CosLon", "SinLon"]
#     test_X_raw.loc[test_X_raw['SM_lat'] < 0,
#                    'Dipole Tilt Angle'] = test_X_raw.loc[test_X_raw['SM_lat'] < 0,
#                                                          'Dipole Tilt Angle'] * (-1)
#     test_X_raw.loc[test_X_raw['SM_lat'] < 0, 'SM_lat'] = test_X_raw.loc[test_X_raw['SM_lat'] < 0, 'SM_lat'] * (-1)
#     return test_X_raw, test_Y_raw


# In[ ]:


station_path = path + '../Input/Gannon_storm/AllStations_Gannon/'
station_file = os.listdir(station_path)
Geospace_path = path + '../Input/Gannon_storm/Geospace_v2/'
Geospace_stations= sorted([x.split('.txt')[0] for x in os.listdir(Geospace_path)])
My_stations = sorted([x.split('.pkl')[0] for x in station_file])
station_to_evaluate = sorted(list(set(Geospace_stations) & set(My_stations)))
station_to_evaluate_file = sorted([x for x in station_file if (x.split('.pkl')[0] in station_to_evaluate)])


# In[ ]:


# geospace_missing = set(My_stations).difference(set(Geospace_stations))
# with open("J:/Paper_deltaB/missing_Stations.txt", "w") as f:
#     for item in geospace_missing:
#         f.write(str(item) + "\n")


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


if make_TestSet3:
    threholds = [50, 100, 200, 300, 400]
    colnames = ["HSS_L%d" % (i+1) for i in range(len(threholds))] + ["Station", "MAE", "SignRate", "Method"]
    for j in range(len(May10_storm)):
        storm_onset = May10_storm[j]
        storm_start = storm_onset - datetime.timedelta(hours = lower)
        storm_end = storm_onset + datetime.timedelta(hours = upper)
        storm_folder_path = res_path + storm_onset.strftime("%Y_%m_%d")
        os.makedirs(storm_folder_path, exist_ok=True)
        HSS_df = []
        for i in track(range(len(station_to_evaluate_file))):
            # Geospace model predictions: --------------------------------------------------------------------------
            GeospaceSim_file = Geospace_path + station_to_evaluate_file[i].split('.pkl')[0] + '.txt'
            df = pd.read_csv(GeospaceSim_file, skiprows=4, delimiter='\s+')
            ## Apply the function to each row in the DataFrame
            df['Datetime'] = df.apply(convert_to_datetime, axis=1)
            df.set_index('Datetime', inplace=True)
            if QoIs == "dBH":
                df['%sSim' % QoIs] = np.sqrt(df['B_EastGeomag']**2 + df['B_NorthGeomag']**2)
            elif QoIs == "dBE":
                df['%sSim' % QoIs] = df['B_EastGeomag']
            else:
                df['%sSim' % QoIs] = df['B_NorthGeomag']
            storm_Geospace = df[storm_start:storm_end].drop_duplicates()
            if len(storm_Geospace) == 0:
                continue
            else:
                # Deep GP model predictions: --------------------------------------------------------------------------
                file_path = station_path + station_to_evaluate_file[i]
                station_data = pd.read_pickle(file_path)
                station_data = station_data.dropna()
                station_data.index = station_data['Time']
                station_data = station_data[[QoIs, "SM_lon", "SM_lat"]]
                storm_X, storm_Y = test_general_matchGeospace(file_path, OMNI, QoIs, storm_Geospace.index)
                Joint_index = storm_Y.index.intersection(storm_Geospace.index)
                if multihour_ahead:
                    ## further consider the data availability of observation persitence
                    Joint_index = station_data.index.intersection(Joint_index-deltaT-deltaT_SW)+deltaT+deltaT_SW
                    observation_persistence = station_data.loc[Joint_index-deltaT-deltaT_SW][QoIs].to_numpy().reshape(-1,1)
                else:
                    Joint_index = station_data.index.intersection(Joint_index-deltaT_SW)+deltaT_SW
                    observation_persistence = station_data.loc[Joint_index-deltaT_SW][QoIs].to_numpy().reshape(-1,1)
                storm_X = storm_X.loc[Joint_index]
                storm_Y = storm_Y.loc[Joint_index]
                if not storm_X.shape[0] == 0:
                    if multihour_ahead:
                        storm_Y_OG, pred_Y_OG, pred_Y_OG_lr, pred_Y_OG_up = DGP_pred(storm_X,storm_Y,model)
                        storm_Y_OG_p, pred_Y_OG_p, pred_Y_OG_lr_p, pred_Y_OG_up_p = DGP_pred(storm_X,storm_Y,model_p)
                    else:
                        storm_Y_OG, pred_Y_OG, pred_Y_OG_lr, pred_Y_OG_up = DGP_pred(storm_X,storm_Y,model)
#                     ## standardize data
#                     storm_Y = scipy.stats.yeojohnson(storm_Y.to_numpy().reshape(-1,1), lmbda=yeojohnson_lmbda)
#                     storm_X = scalerX.transform(storm_X.to_numpy())
#                     storm_Y = scalerY.transform(storm_Y)
#                     storm_X = torch.from_numpy(storm_X).type(torch.float32).to(get_device())
#                     storm_Y = torch.from_numpy(storm_Y).type(torch.float32).to(get_device())
#                     storm_Y = storm_Y.reshape(storm_Y.shape[0])
#                     storm_dataset = SuperMAGDataset_test(storm_X, storm_Y)
#                     storm_loader = DataLoader(storm_dataset, batch_size=4096)

#                     with torch.no_grad():
#                         pred_means, pred_var, lls = model.predict(storm_loader)
#                         sigma = sigma_GMM(pred_means, pred_var)
#                         qnorm_975 = scipy.stats.norm.ppf(0.975, loc=0, scale=1)
#                         pred_Y_DGP = pred_means.mean(0).cpu().numpy().reshape(-1,1)
#                         storm_Y_OG = scalerY.inverse_transform(storm_loader.dataset.Y.cpu().numpy().reshape(-1,1))
#                         pred_Y_OG = scalerY.inverse_transform(pred_Y_DGP)
#                         storm_Y_OG = np.array([invert_yeojhonson(x, yeojohnson_lmbda) for x in storm_Y_OG])
#                         pred_Y_OG = np.array([invert_yeojhonson(x, yeojohnson_lmbda) for x in pred_Y_OG])
#         #                 storm_Y_OG = scipy.special.inv_boxcox(storm_Y_OG, boxcox_lmbda, out=None)
#         #                 pred_Y_OG = scipy.special.inv_boxcox(pred_Y_OG, boxcox_lmbda, out=None)

#                         pred_Y_DGP_lr = pred_Y_DGP - qnorm_975*sigma.reshape(-1,1)
#                         pred_Y_DGP_up = pred_Y_DGP + qnorm_975*sigma.reshape(-1,1)
#                         pred_Y_OG_lr = np.array([invert_yeojhonson(x, yeojohnson_lmbda)
#                                                  for x in scalerY.inverse_transform(pred_Y_DGP_lr)]).flatten()
#                         pred_Y_OG_up = np.array([invert_yeojhonson(x, yeojohnson_lmbda)
#                                                  for x in scalerY.inverse_transform(pred_Y_DGP_up)]).flatten()

#         #                 pred_Y_OG_lr = scipy.special.inv_boxcox(scalerY.inverse_transform(pred_Y_DGP_lr),
#         #                                                         boxcox_lmbda, out=None).flatten()
#         #                 pred_Y_OG_up = scipy.special.inv_boxcox(scalerY.inverse_transform(pred_Y_DGP_up),
#         #                                                         boxcox_lmbda, out=None).flatten()
                    # HSS calculation: -----------------------------------------------------------------------
                    ## Deep GP HSS calculation
                    HSS = []
                    for threshold in threholds:
                        HSS.append(compute_hss(np.sign(storm_Y_OG) * storm_Y_OG,
                                               np.sign(storm_Y_OG) * pred_Y_OG, threshold).iloc[0])
                    HSS_pd = pd.DataFrame(HSS).T
                    HSS_pd["Station"] = station_to_evaluate_file[i][0:3]
                    MAE = mean_absolute_error(storm_Y_OG, pred_Y_OG)
                    HSS_pd["MAE"] = int(MAE)
                    HSS_pd["SignRate"] = np.mean(np.sign(storm_Y_OG) == np.sign(pred_Y_OG))
                    HSS_pd["Method"] = "GeoDGP"
                    HSS_pd.columns = colnames
                    HSS_df.append(HSS_pd)
                    ## Geospace HSS calculation
                    HSS = []
                    Geospace_Y_OG = storm_Geospace.loc[Joint_index]['%sSim' % QoIs].to_numpy().reshape(-1,1)
                    for threshold in threholds:
                        HSS.append(compute_hss(np.sign(storm_Y_OG) * storm_Y_OG,
                                               np.sign(storm_Y_OG) * Geospace_Y_OG,
                                               threshold).iloc[0])
                    HSS_pd = pd.DataFrame(HSS).T
                    HSS_pd["Station"] = station_to_evaluate_file[i][0:3]
                    MAE = mean_absolute_error(storm_Y_OG,
                                              storm_Geospace.loc[Joint_index]['%sSim' % QoIs].to_numpy().reshape(-1,1))
                    HSS_pd["MAE"] = int(MAE)
                    HSS_pd["SignRate"] = np.mean(np.sign(storm_Y_OG) ==  np.sign(Geospace_Y_OG))
                    HSS_pd["Method"] = "Geospace"
                    HSS_pd.columns = colnames
                    HSS_df.append(HSS_pd)
                    ## Observation Persistence HSS calculation
                    HSS = []
                    for threshold in threholds:
                        HSS.append(compute_hss(np.sign(storm_Y_OG) * storm_Y_OG,
                                               np.sign(storm_Y_OG) * observation_persistence,
                                               threshold).iloc[0])
                    HSS_pd = pd.DataFrame(HSS).T
                    HSS_pd["Station"] = station_to_evaluate_file[i][0:3]
                    MAE = mean_absolute_error(storm_Y_OG, observation_persistence)
                    HSS_pd["MAE"] = int(MAE)
                    HSS_pd["SignRate"] = np.mean(np.sign(storm_Y_OG) == np.sign(observation_persistence))
                    HSS_pd["Method"] = "Observation_persistence"
                    HSS_pd.columns = colnames
                    HSS_df.append(HSS_pd)
                    if multihour_ahead:
                        ## model persistence HSS calculation
                        HSS = []
                        for threshold in threholds:
                            HSS.append(compute_hss(np.sign(storm_Y_OG) * storm_Y_OG,
                                                   np.sign(storm_Y_OG) * pred_Y_OG_p,
                                                   threshold).iloc[0])
                        HSS_pd = pd.DataFrame(HSS).T
                        HSS_pd["Station"] = station_to_evaluate_file[i][0:3]
                        MAE = mean_absolute_error(storm_Y_OG, pred_Y_OG_p)
                        HSS_pd["MAE"] = int(MAE)
                        HSS_pd["SignRate"] = np.mean(np.sign(storm_Y_OG) == np.sign(pred_Y_OG_p))
                        HSS_pd["Method"] = "GeoDGP_persistence"
                        HSS_pd.columns = colnames
                        HSS_df.append(HSS_pd)
                    # visualization: -----------------------------------------------------------------------
                    if plot_image == True:
                        time_idx = storm_Geospace.loc[Joint_index]['%sSim' % QoIs].resample('1T').median().index
                        plt.rcParams["figure.figsize"] = [24,8]
                        plt.rcParams.update({'font.size': 25})

                        ## Geospace
                        plt.plot(time_idx, storm_Geospace.loc[Joint_index]['%sSim' % QoIs].resample('1T').median(),
                                 color = "royalblue", label = "Geospace")
                        ## Deep GP
                        pred_Y_OG_df = pd.DataFrame(pred_Y_OG)
                        pred_Y_OG_df.index = Joint_index
                        pred_Y_OG_df.columns = ['%s' % QoIs]
                        plt.plot(time_idx, pred_Y_OG_df['%s' % QoIs].resample('1T').median(),
                                 color= "red", label="GeoDGP")
                        ## Observation
                        storm_Y_OG_df = pd.DataFrame(storm_Y_OG)
                        storm_Y_OG_df.index = Joint_index
                        storm_Y_OG_df.columns = ['%s' % QoIs]
                        plt.plot(time_idx, storm_Y_OG_df['%s' % QoIs].resample('1T').median(),
                                 color= "black", label="Obs")
                        plt.xticks(rotation=15)
                        ## prediction interval of Deep GP
                        pred_Y_OG_lr_df = pd.DataFrame(pred_Y_OG_lr)
                        pred_Y_OG_lr_df.index = Joint_index
                        pred_Y_OG_lr_df.columns = ['%s' % QoIs]
                        pred_Y_OG_up_df = pd.DataFrame(pred_Y_OG_up)
                        pred_Y_OG_up_df.index = Joint_index
                        pred_Y_OG_up_df.columns = ['%s' % QoIs]
                        if QoIs == "dBH":
                            plt.fill_between(time_idx,
                                         np.clip(pred_Y_OG_lr_df['%s' % QoIs].resample('1T').median(),a_min=0, a_max=3000),
                                         pred_Y_OG_up_df['%s' % QoIs].resample('1T').median(),
                                         color='lightgrey', alpha=.7)
                        else:
                            plt.fill_between(time_idx,
                                         pred_Y_OG_lr_df['%s' % QoIs].resample('1T').median(),
                                         pred_Y_OG_up_df['%s' % QoIs].resample('1T').median(),
                                         color='lightgrey', alpha=.7)
                        ## refinement
                        plt.ylabel("%s [nT]" % QoIs, fontsize = 40)
                        plt.legend(loc="upper left")
                        ax = plt.gca()
                        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
                        plt.text(0.82, .95, station_to_evaluate_file[i][0:3], ha='left', va='top', transform=ax.transAxes,
                                 fontsize=40)
                        plt.savefig(storm_folder_path+"/"+ station_to_evaluate_file[i][0:3]+".png",
                                    bbox_inches='tight')
                        plt.close()
        if len(HSS_df) >0:
            HSS_df = pd.concat(HSS_df, axis=0)
            first_column = HSS_df.pop('Station') 

            # insert column using insert(position,column_name, 
            # first_column) function 
            HSS_df.insert(0, 'Station', first_column) 
            HSS_df.to_csv(res_path+"HSS_%s.csv" % May10_storm[j].strftime("%Y_%m_%d"), index=False)
        else:
            print("Storm %s data not available" % May10_storm[j].strftime("%Y_%m_%d"))


# In[ ]:


if make_TestSet3_figure:
    plt.tight_layout()
    plt.rcParams.update({'font.size': 17})
    fig, axes = plt.subplots(3, 2, figsize=(24, 13))
    for j in range(len(May10_storm)):
        storm_onset = May10_storm[j]
        storm_start = storm_onset - datetime.timedelta(hours = lower)
        storm_end = storm_onset + datetime.timedelta(hours = upper)
        storm_folder_path = res_path + storm_onset.strftime("%Y_%m_%d")
        os.makedirs(storm_folder_path, exist_ok=True)
        HSS_df = []
#         plt.rcParams["figure.figsize"] = [15,10]
        for i in track(range(len(station_to_evaluate_file))):
            if station_to_evaluate_file[i][0:3] in station_to_visualize:
                # Geospace model predictions: --------------------------------------------------------------------------
                GeospaceSim_file = Geospace_path + station_to_evaluate_file[i].split('.pkl')[0] + '.txt'
                df = pd.read_csv(GeospaceSim_file, skiprows=4, delimiter='\s+')
                ## Apply the function to each row in the DataFrame
                df['Datetime'] = df.apply(convert_to_datetime, axis=1)
                df.set_index('Datetime', inplace=True)
                if QoIs == "dBH":
                    df['%sSim' % QoIs] = np.sqrt(df['B_EastGeomag']**2 + df['B_NorthGeomag']**2)
                elif QoIs == "dBE":
                    df['%sSim' % QoIs] = df['B_EastGeomag']
                else:
                    df['%sSim' % QoIs] = df['B_NorthGeomag']
                storm_Geospace = df[storm_start:storm_end].drop_duplicates()
                if len(storm_Geospace) == 0:
                    continue
                else:
                    # Deep GP model predictions: --------------------------------------------------------------------------
                    file_path = station_path + station_to_evaluate_file[i]
                    storm_X, storm_Y = test_general_matchGeospace(file_path, OMNI, QoIs, storm_Geospace.index)
                    Joint_index = storm_Y.index.intersection(storm_Geospace.index)
                    storm_X = storm_X.loc[Joint_index]
                    storm_Y = storm_Y.loc[Joint_index]
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
                            pred_means, pred_var, lls = model.predict(storm_loader)
                            sigma = sigma_GMM(pred_means, pred_var)
                            qnorm_975 = scipy.stats.norm.ppf(0.975, loc=0, scale=1)
                            pred_Y_DGP = pred_means.mean(0).cpu().numpy().reshape(-1,1)
                            storm_Y_OG = scalerY.inverse_transform(storm_loader.dataset.Y.cpu().numpy().reshape(-1,1))
                            pred_Y_OG = scalerY.inverse_transform(pred_Y_DGP)
                            storm_Y_OG = np.array([invert_yeojhonson(x, yeojohnson_lmbda) for x in storm_Y_OG])
                            pred_Y_OG = np.array([invert_yeojhonson(x, yeojohnson_lmbda) for x in pred_Y_OG])
            #                 storm_Y_OG = scipy.special.inv_boxcox(storm_Y_OG, boxcox_lmbda, out=None)
            #                 pred_Y_OG = scipy.special.inv_boxcox(pred_Y_OG, boxcox_lmbda, out=None)

                            pred_Y_DGP_lr = pred_Y_DGP - qnorm_975*sigma.reshape(-1,1)
                            pred_Y_DGP_up = pred_Y_DGP + qnorm_975*sigma.reshape(-1,1)
                            pred_Y_OG_lr = np.array([invert_yeojhonson(x, yeojohnson_lmbda)
                                                     for x in scalerY.inverse_transform(pred_Y_DGP_lr)]).flatten()
                            pred_Y_OG_up = np.array([invert_yeojhonson(x, yeojohnson_lmbda)
                                                     for x in scalerY.inverse_transform(pred_Y_DGP_up)]).flatten()

            #                 pred_Y_OG_lr = scipy.special.inv_boxcox(scalerY.inverse_transform(pred_Y_DGP_lr),
            #                                                         boxcox_lmbda, out=None).flatten()
            #                 pred_Y_OG_up = scipy.special.inv_boxcox(scalerY.inverse_transform(pred_Y_DGP_up),
            #                                                         boxcox_lmbda, out=None).flatten()
                        # visualization: -----------------------------------------------------------------------
                        index = np.where(np.array(station_to_visualize) == station_to_evaluate_file[i][0:3])[0][0]
                        time_idx = storm_Geospace.loc[Joint_index]['%sSim' % QoIs].resample('1T').median().index
                        time_idx = time_idx[time_idx < datetime.datetime(2024, 5, 12, 9)]
                        ax = axes.flat[index]
                        ## Geospace
                        ax.plot(time_idx, storm_Geospace.loc[Joint_index]['%sSim' % QoIs].resample('1T').median().loc[time_idx],
                                 color = "mediumblue", label = 'Geospace ' + r'$(T_{S})$', linewidth=1.2)
                        ## Deep GP
                        pred_Y_OG_df = pd.DataFrame(pred_Y_OG)
                        pred_Y_OG_df.index = Joint_index
                        pred_Y_OG_df.columns = ['%s' % QoIs]
                        ax.plot(time_idx, pred_Y_OG_df['%s' % QoIs].resample('1T').median().loc[time_idx],
                                 color= "red", label='GeoDGP ' + r'$(T_{S})$', linewidth=0.8)
                        ## Observation
                        storm_Y_OG_df = pd.DataFrame(storm_Y_OG)
                        storm_Y_OG_df.index = Joint_index
                        storm_Y_OG_df.columns = ['%s' % QoIs]
                        ax.plot(time_idx, storm_Y_OG_df['%s' % QoIs].resample('1T').median().loc[time_idx],
                                 color= "black", label="Station", linewidth=0.8)
                        ax.tick_params(axis='x', rotation=10)
                        ax.xaxis.set_major_locator(mdates.HourLocator(interval=9))
                        ## prediction interval of Deep GP
                        pred_Y_OG_lr_df = pd.DataFrame(pred_Y_OG_lr)
                        pred_Y_OG_lr_df.index = Joint_index
                        pred_Y_OG_lr_df.columns = ['%s' % QoIs]
                        pred_Y_OG_up_df = pd.DataFrame(pred_Y_OG_up)
                        pred_Y_OG_up_df.index = Joint_index
                        pred_Y_OG_up_df.columns = ['%s' % QoIs]
                        if QoIs == "dBH":
                            ax.fill_between(time_idx,
                                         np.clip(pred_Y_OG_lr_df['%s' % QoIs].resample('1T').median().loc[time_idx],
                                                 a_min=0, a_max=3000),
                                         pred_Y_OG_up_df['%s' % QoIs].resample('1T').median().loc[time_idx],
                                         color='lightgrey', alpha=.7)
                        else:
                            ax.fill_between(time_idx,
                                         pred_Y_OG_lr_df['%s' % QoIs].resample('1T').median().loc[time_idx],
                                         pred_Y_OG_up_df['%s' % QoIs].resample('1T').median().loc[time_idx],
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
                            plt.savefig(path_prefix+"/Paper_deltaB/figure/TestSet3/TestSet3_stations_%s.png" % QoIs,
                                        bbox_inches='tight')
                            plt.close()


# In[ ]:


# ## test mode
# station_to_evaluate_FFTfile = sorted([x for x in station_file if (x.split('.pkl')[0] in station_to_evaluate_FFT)])
# ## --------------------------------------------- save for station ABK only ---------------------------------------
# if QoIs == "dBH_dt":
#     threhold_list = [18, 42, 66, 90]
# elif (QoIs == "dBH") or (QoIs == "dBN"):
#     threhold_list = [50, 200, 300, 400]
# else:
#     threhold_list = [50, 100, 200, 300]
# j=0
# storm_onset = May10_storm[j]
# storm_start = storm_onset - datetime.timedelta(hours = lower)
# storm_end = storm_onset + datetime.timedelta(hours = upper)
# storm_folder_path = res_path + storm_onset.strftime("%Y_%m_%d")
# os.makedirs(storm_folder_path, exist_ok=True)
# HSS_df = []
# i=0
# # Geospace model predictions: --------------------------------------------------------------------------
# GeospaceSim_file = Geospace_path + station_to_evaluate_FFTfile[i].split('.pkl')[0] + '.txt'
# df = pd.read_csv(GeospaceSim_file, skiprows=4, delimiter='\s+')
# ## Apply the function to each row in the DataFrame
# df['Datetime'] = df.apply(convert_to_datetime, axis=1)
# df.set_index('Datetime', inplace=True)
# if QoIs == "dBH":
#     df['%sSim' % QoIs] = np.sqrt(df['B_EastGeomag']**2 + df['B_NorthGeomag']**2)
# elif QoIs == "dBE":
#     df['%sSim' % QoIs] = df['B_EastGeomag']
# else:
#     df['%sSim' % QoIs] = df['B_NorthGeomag']
# storm_Geospace = df[storm_start:storm_end].drop_duplicates()
# # if len(storm_Geospace) == 0:
# #     continue
# # else:
# # Deep GP model predictions: --------------------------------------------------------------------------
# file_path = station_path + station_to_evaluate_FFTfile[i]
# storm_X, storm_Y = test_general_matchGeospace(file_path, OMNI, QoIs, storm_Geospace.index)
# Joint_index = storm_Y.index.intersection(storm_Geospace.index)
# storm_X = storm_X.loc[Joint_index]
# storm_Y = storm_Y.loc[Joint_index]
# if not storm_X.shape[0] == 0:
#     ## standardize data
#     storm_X = scalerX.transform(storm_X.to_numpy())
# #     storm_Y = scalerY.transform(storm_Y.reshape(1, -1))
#     storm_X = torch.from_numpy(storm_X).type(torch.float32).to(get_device())
# #     storm_Y = torch.from_numpy(storm_Y).type(torch.float32).to(get_device())
# #     storm_Y = storm_Y.reshape(storm_Y.shape[0])
# #     storm_dataset = SuperMAGDataset_test(storm_X, storm_Y)
# #     storm_loader = DataLoader(storm_dataset, batch_size=4096)


# In[ ]:


# num_realizations=200
# num_samples = 1
# samples = []
# with torch.no_grad():
#     with gpytorch.settings.num_likelihood_samples(num_samples):
#         for _ in range(num_realizations):
#             realization = model(storm_X).rsample().squeeze(0)
#             samples.append(scalerY.inverse_transform(realization.cpu().numpy().reshape(-1,1)).reshape(-1))
# samples = np.stack(samples, axis=0)


# In[ ]:


# pred_mean = np.mean(samples, axis=0)
# lr_bound = np.percentile(samples, 2.5, axis=0)
# upr_bound = np.percentile(samples, 97.5, axis=0)

# time_steps = np.arange(samples.shape[1])  # X-axis: Time steps

# plt.figure(figsize=(10, 5))

# # Plot mean time series
# plt.plot(time_steps, pred_mean, color='red', label="Mean Prediction")
# plt.plot(time_steps, storm_Y, label='Station ABK', color='black')
# # Fill between the quantile bounds to show uncertainty
# plt.fill_between(time_steps, lr_bound, upr_bound, color='gray', alpha=0.5, label="95% Confidence Interval", edgecolor='white')

# plt.xlabel("Time Steps")
# plt.ylabel("Value")
# plt.title("Mean Prediction with 95% Confidence Interval")
# plt.legend()
# plt.show()


# In[ ]:


# # k = 0
# # plt.plot(range(samples.shape[1]), storm_Y)
# # plt.plot(range(samples.shape[1]), samples[k, :])
# # Create subplots
# fig, axes = plt.subplots(nrows=num_realizations, ncols=1, figsize=(10, 3 * num_realizations), sharex=True)

# # If only one subplot, wrap it in a list for iteration
# if num_realizations == 1:
#     axes = [axes]

# # Loop through each realization and plot
# for k in range(num_realizations):
#     axes[k].plot(range(samples.shape[1]), storm_Y, label="True", color='black', linewidth=2)
#     axes[k].plot(range(samples.shape[1]), samples[k, :], label=f"Sample {k+1}", color='red')
#     axes[k].legend()
#     axes[k].set_title(f"Realization {k+1}")

# # Improve spacing
# plt.xlabel("Time Steps")
# plt.tight_layout()
# plt.show()


# In[ ]:


# import time
# start = time.time()
# if QoIs == 'dBN':
#     column_name = 'B_NorthGeomag'
# if QoIs == 'dBE':
#     column_name = 'B_EastGeomag'
# if QoIs == 'dBH':
#     column_name = 'dBH'
# num_realizations=100
# num_samples = 1
# station_to_evaluate_FFTfile = sorted([x for x in station_file if (x.split('.pkl')[0] in station_to_evaluate_FFT)])
# ## --------------------------------------------- save for station ABK only ---------------------------------------
# if QoIs == "dBH_dt":
#     threhold_list = [18, 42, 66, 90]
# elif (QoIs == "dBH") or (QoIs == "dBN"):
#     threhold_list = [50, 200, 300, 400]
# else:
#     threhold_list = [50, 100, 200, 300]
# j=0
# storm_onset = May10_storm[j]
# storm_start = storm_onset - datetime.timedelta(hours = lower)
# storm_end = storm_onset + datetime.timedelta(hours = upper)
# storm_folder_path = res_path + storm_onset.strftime("%Y_%m_%d")
# os.makedirs(storm_folder_path, exist_ok=True)
# HSS_df = []
# i=0
# # Geospace model predictions: --------------------------------------------------------------------------
# GeospaceSim_file = Geospace_path + station_to_evaluate_FFTfile[i].split('.pkl')[0] + '.txt'
# df = pd.read_csv(GeospaceSim_file, skiprows=4, delimiter='\s+')
# ## Apply the function to each row in the DataFrame
# df['Datetime'] = df.apply(convert_to_datetime, axis=1)
# df.set_index('Datetime', inplace=True)
# if QoIs == "dBH":
#     df['%sSim' % QoIs] = np.sqrt(df['B_EastGeomag']**2 + df['B_NorthGeomag']**2)
# elif QoIs == "dBE":
#     df['%sSim' % QoIs] = df['B_EastGeomag']
# else:
#     df['%sSim' % QoIs] = df['B_NorthGeomag']
# storm_Geospace = df[storm_start:storm_end].drop_duplicates()
# # Deep GP model predictions: --------------------------------------------------------------------------
# file_path = station_path + station_to_evaluate_FFTfile[i]
# storm_X, storm_Y = test_general_matchGeospace(file_path, OMNI, QoIs, storm_Geospace.index)
# Joint_index = storm_Y.index.intersection(storm_Geospace.index)
# storm_X = storm_X.loc[Joint_index]
# storm_Y = storm_Y.loc[Joint_index]
# if not storm_X.shape[0] == 0:
#     ## standardize data
#     storm_X = scalerX.transform(storm_X.to_numpy())
# #     storm_Y = scalerY.transform(storm_Y.reshape(1, -1))
#     storm_X = torch.from_numpy(storm_X).type(torch.float32).to(get_device())
#     samples = []
#     with torch.no_grad():
#         with gpytorch.settings.num_likelihood_samples(num_samples):
#             for real_num in range(num_realizations):
#                 realization = model(storm_X).rsample().squeeze(0)
#                 realization_OG = scalerY.inverse_transform(realization.cpu().numpy().reshape(-1,1)).reshape(-1)
#                 samples.append(pd.DataFrame({column_name: realization_OG,
#                                             'realization': real_num},
#                              index = Joint_index))
#     samples = pd.concat(samples, axis=0)
#     samples['Year'] = [time.year for time in samples.index]
#     samples['Month'] = [time.month for time in samples.index]
#     samples['Day'] = [time.day for time in samples.index]
#     samples['Hour'] = [time.hour for time in samples.index]
#     samples['Min'] = [time.minute for time in samples.index]
#     samples['Sec'] = [time.second for time in samples.index]
# end = time.time()
# end-start


# In[ ]:


# os.makedirs(path_prefix+"/Paper_deltaB/data/test_station/FFT_withUQ/QoIs/", exist_ok=True)
# if QoIs == 'dBN':
#     column_name = 'B_NorthGeomag'
# if QoIs == 'dBE':
#     column_name = 'B_EastGeomag'
# if QoIs == 'dBH':
#     column_name = 'dBH'
# num_realizations=200
# num_samples = 1
# station_to_evaluate_FFTfile = sorted([x for x in station_file if (x.split('.pkl')[0] in station_to_evaluate_FFT)])
# ## --------------------------------------------- save for station ABK only ---------------------------------------
# if QoIs == "dBH_dt":
#     threhold_list = [18, 42, 66, 90]
# elif (QoIs == "dBH") or (QoIs == "dBN"):
#     threhold_list = [50, 200, 300, 400]
# else:
#     threhold_list = [50, 100, 200, 300]
# for j in range(len(May10_storm)):
#     storm_onset = May10_storm[j]
#     storm_start = storm_onset - datetime.timedelta(hours = lower)
#     storm_end = storm_onset + datetime.timedelta(hours = upper)
#     storm_folder_path = res_path + storm_onset.strftime("%Y_%m_%d")
#     os.makedirs(storm_folder_path, exist_ok=True)
#     HSS_df = []
#     i=0
#     # Geospace model predictions: --------------------------------------------------------------------------
#     GeospaceSim_file = Geospace_path + station_to_evaluate_FFTfile[i].split('.pkl')[0] + '.txt'
#     df = pd.read_csv(GeospaceSim_file, skiprows=4, delimiter='\s+')
#     ## Apply the function to each row in the DataFrame
#     df['Datetime'] = df.apply(convert_to_datetime, axis=1)
#     df.set_index('Datetime', inplace=True)
#     if QoIs == "dBH":
#         df['%sSim' % QoIs] = np.sqrt(df['B_EastGeomag']**2 + df['B_NorthGeomag']**2)
#     elif QoIs == "dBE":
#         df['%sSim' % QoIs] = df['B_EastGeomag']
#     else:
#         df['%sSim' % QoIs] = df['B_NorthGeomag']
#     storm_Geospace = df[storm_start:storm_end].drop_duplicates()
#     if len(storm_Geospace) == 0:
#         continue
#     else:
#         # Deep GP model predictions: --------------------------------------------------------------------------
#         file_path = station_path + station_to_evaluate_FFTfile[i]
#         storm_X, storm_Y = test_general_matchGeospace(file_path, OMNI, QoIs, storm_Geospace.index)
#         Joint_index = storm_Y.index.intersection(storm_Geospace.index)
#         storm_X = storm_X.loc[Joint_index]
#         storm_Y = storm_Y.loc[Joint_index]
#         if not storm_X.shape[0] == 0:
#             ## standardize data
#             storm_X = scalerX.transform(storm_X.to_numpy())
#         #     storm_Y = scalerY.transform(storm_Y.reshape(1, -1))
#             storm_X = torch.from_numpy(storm_X).type(torch.float32).to(get_device())
#             samples = []
#             with torch.no_grad():
#                 with gpytorch.settings.num_likelihood_samples(num_samples):
#                     for real_num in range(num_realizations):
#                         realization = model(storm_X).rsample().squeeze(0)
#                         realization_OG = scalerY.inverse_transform(realization.cpu().numpy().reshape(-1,1)).reshape(-1)
#                         samples.append(pd.DataFrame({column_name: realization_OG,
#                                                     'realization': real_num},
#                                      index = Joint_index))
#             samples = pd.concat(samples, axis=0)


# In[ ]:


# df = pd.read_csv("J:/Paper_deltaB/data/test_station/FFT_withUQ/merged/ABK.csv")
# df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']].rename(columns={
#     'Year': 'year',
#     'Month': 'month',
#     'Day': 'day',
#     'Hour': 'hour'
# })).add(pd.to_timedelta(df['Min'], unit='m') + pd.to_timedelta(df['Sec'], unit='s'))
# pivoted = df.pivot(index='datetime', columns='realization', values='B_EastGeomag')
# # pivoted = samples.pivot_table(index=samples.index, columns='realization', values='B_NorthGeomag')


# In[ ]:


# df_old = pd.read_csv("J:/Paper_deltaB/data/test_station/FFT/ABK_dBE.csv")
# df_old['datetime'] = pd.to_datetime(df_old[['Year', 'Month', 'Day', 'Hour']].rename(columns={
#     'Year': 'year',
#     'Month': 'month',
#     'Day': 'day',
#     'Hour': 'hour'
# })).add(pd.to_timedelta(df_old['Min'], unit='m') + pd.to_timedelta(df_old['Sec'], unit='s'))


# In[ ]:


# # Quantiles across realizations
# q25 = pivoted.quantile(0.025, axis=1)
# q975 = pivoted.quantile(0.975, axis=1)
# mean = pivoted.quantile(0.5, axis=1)
# # Plot
# plt.figure(figsize=(16, 8))
# plt.plot(mean.index, mean, label='New Mean', linestyle='-',color='red')
# plt.plot(mean.index, df_old['B_EastGeomag'], label='old', linestyle='-',color='blue', alpha = 0.5)
# plt.fill_between(q25.index, q25, q975, alpha=0.5, label='Quantile Range', color='gray')
# plt.title('ABK: Quantile Range of B_NorthGeomag Across Realizations', fontsize=20)
# plt.xlabel('Time', fontsize=20)
# plt.ylabel('B_NorthGeomag', fontsize=20)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# In[ ]:


# plt.figure(figsize=(16, 8))
# q25 = pivoted.quantile(0.025, axis=1)
# q975 = pivoted.quantile(0.975, axis=1)
# mean = pivoted.quantile(0.5, axis=1)
# plt.plot(mean.index, mean, label='mean', linestyle='-',color='red')
# plt.fill_between(mean.index, q25, q975, alpha=0.5, label='Quantile Range', color='gray')


# In[ ]:


"""
Preparing QoIs for FFT with uncertainty quantification. We do random sampling to propagate uncertainty in QoIs to
the downstream FFT calculation and geoelectric field integration.
"""
if make_FFTfile:
    os.makedirs(path_prefix+"/Paper_deltaB/data/test_station/FFT_withUQ/QoIs/", exist_ok=True)
    if QoIs == 'dBN':
        column_name = 'B_NorthGeomag'
    if QoIs == 'dBE':
        column_name = 'B_EastGeomag'
    if QoIs == 'dBH':
        column_name = 'dBH'
    num_realizations=200
    num_samples = 1
    station_to_evaluate_FFTfile = sorted([x for x in station_file if (x.split('.pkl')[0] in station_to_evaluate_FFT)])
    ## --------------------------------------------- save for station ABK only ---------------------------------------
    if QoIs == "dBH_dt":
        threhold_list = [18, 42, 66, 90]
    elif (QoIs == "dBH") or (QoIs == "dBN"):
        threhold_list = [50, 200, 300, 400]
    else:
        threhold_list = [50, 100, 200, 300]
    for j in range(len(May10_storm)):
        storm_onset = May10_storm[j]
        storm_start = storm_onset - datetime.timedelta(hours = lower)
        storm_end = storm_onset + datetime.timedelta(hours = upper)
        storm_folder_path = res_path + storm_onset.strftime("%Y_%m_%d")
        os.makedirs(storm_folder_path, exist_ok=True)
        HSS_df = []
        for i in track(range(len(station_to_evaluate_FFTfile))):
            # Geospace model predictions: --------------------------------------------------------------------------
            GeospaceSim_file = Geospace_path + station_to_evaluate_FFTfile[i].split('.pkl')[0] + '.txt'
            df = pd.read_csv(GeospaceSim_file, skiprows=4, delimiter='\s+')
            ## Apply the function to each row in the DataFrame
            df['Datetime'] = df.apply(convert_to_datetime, axis=1)
            df.set_index('Datetime', inplace=True)
            if QoIs == "dBH":
                df['%sSim' % QoIs] = np.sqrt(df['B_EastGeomag']**2 + df['B_NorthGeomag']**2)
            elif QoIs == "dBE":
                df['%sSim' % QoIs] = df['B_EastGeomag']
            else:
                df['%sSim' % QoIs] = df['B_NorthGeomag']
            storm_Geospace = df[storm_start:storm_end].drop_duplicates()
            if len(storm_Geospace) == 0:
                continue
            else:
                # Deep GP model predictions: --------------------------------------------------------------------------
                file_path = station_path + station_to_evaluate_FFTfile[i]
                storm_X, storm_Y = test_general_matchGeospace(file_path, OMNI, QoIs, storm_Geospace.index)
                Joint_index = storm_Y.index.intersection(storm_Geospace.index)
                storm_X = storm_X.loc[Joint_index]
                storm_Y = storm_Y.loc[Joint_index]
                if not storm_X.shape[0] == 0:
                    ## standardize data
                    storm_X = scalerX.transform(storm_X.to_numpy())
                #     storm_Y = scalerY.transform(storm_Y.reshape(1, -1))
                    storm_X = torch.from_numpy(storm_X).type(torch.float32).to(get_device())
                    samples = []
                    with torch.no_grad():
                        with gpytorch.settings.num_likelihood_samples(num_samples):
                            for real_num in range(num_realizations):
                                realization = model(storm_X).rsample().squeeze(0)
                                realization_OG = scalerY.inverse_transform(realization.cpu().numpy().reshape(-1,1)).reshape(-1)
                                samples.append(pd.DataFrame({column_name: realization_OG,
                                                            'realization': real_num},
                                             index = Joint_index))
                    samples = pd.concat(samples, axis=0)
                    samples['Year'] = [time.year for time in samples.index]
                    samples['Month'] = [time.month for time in samples.index]
                    samples['Day'] = [time.day for time in samples.index]
                    samples['Hour'] = [time.hour for time in samples.index]
                    samples['Min'] = [time.minute for time in samples.index]
                    samples['Sec'] = [time.second for time in samples.index]
                    samples.to_csv(path_prefix+"/Paper_deltaB/data/test_station/FFT_withUQ/QoIs/" + 
                                        station_to_evaluate_FFTfile[i][0:3]+'_%s.csv' % QoIs, index=False)
#                     pred_Y_OG_df = pd.DataFrame(pred_Y_OG)
#                     pred_Y_OG_df.index = Joint_index
#                     if QoIs == 'dBN':
#                         pred_Y_OG_df.columns = ['B_NorthGeomag']
#                     if QoIs == 'dBE':
#                         pred_Y_OG_df.columns = ['B_EastGeomag']
#                     if QoIs == 'dBH':
#                         pred_Y_OG_df.columns = ['dBH']
#                         storm_Y_OG_df = pd.DataFrame(storm_Y_OG)
#                         storm_Y_OG_df.index = Joint_index
#                         storm_Y_OG_df.columns = ['%s' % QoIs]
#                         storm_Y_OG_df.to_csv(path_prefix+"/Paper_deltaB/data/test_station/FFT/" + 
#                                             station_to_evaluate_FFTfile[i][0:3]+'_Obs%s.csv' % QoIs, index=False)
#                     pred_Y_OG_df['Year'] = [time.year for time in pred_Y_OG_df.index]
#                     pred_Y_OG_df['Month'] = [time.month for time in pred_Y_OG_df.index]
#                     pred_Y_OG_df['Day'] = [time.day for time in pred_Y_OG_df.index]
#                     pred_Y_OG_df['Hour'] = [time.hour for time in pred_Y_OG_df.index]
#                     pred_Y_OG_df['Min'] = [time.minute for time in pred_Y_OG_df.index]
#                     pred_Y_OG_df['Sec'] = [time.second for time in pred_Y_OG_df.index]
#                     pred_Y_OG_df.to_csv(path_prefix+"/Paper_deltaB/data/test_station/FFT_withUQ/" + 
#                                         station_to_evaluate_FFTfile[i][0:3]+'_%s.csv' % QoIs, index=False)


# In[ ]:


"""
FFT without uncertainty quantification
Deprecated: This chunk is outdated and will be removed in future versions.
"""
# if make_FFTfile:
#     station_to_evaluate_FFTfile = sorted([x for x in station_file if (x.split('.pkl')[0] in station_to_evaluate_FFT)])
#     ## --------------------------------------------- save for station ABK only ---------------------------------------
#     if QoIs == "dBH_dt":
#         threhold_list = [18, 42, 66, 90]
#     elif (QoIs == "dBH") or (QoIs == "dBN"):
#         threhold_list = [50, 200, 300, 400]
#     else:
#         threhold_list = [50, 100, 200, 300]
#     for j in range(len(May10_storm)):
#         storm_onset = May10_storm[j]
#         storm_start = storm_onset - datetime.timedelta(hours = lower)
#         storm_end = storm_onset + datetime.timedelta(hours = upper)
#         storm_folder_path = res_path + storm_onset.strftime("%Y_%m_%d")
#         os.makedirs(storm_folder_path, exist_ok=True)
#         HSS_df = []
#         for i in track(range(len(station_to_evaluate_FFTfile))):
#             # Geospace model predictions: --------------------------------------------------------------------------
#             GeospaceSim_file = Geospace_path + station_to_evaluate_FFTfile[i].split('.pkl')[0] + '.txt'
#             df = pd.read_csv(GeospaceSim_file, skiprows=4, delimiter='\s+')
#             ## Apply the function to each row in the DataFrame
#             df['Datetime'] = df.apply(convert_to_datetime, axis=1)
#             df.set_index('Datetime', inplace=True)
#             if QoIs == "dBH":
#                 df['%sSim' % QoIs] = np.sqrt(df['B_EastGeomag']**2 + df['B_NorthGeomag']**2)
#             elif QoIs == "dBE":
#                 df['%sSim' % QoIs] = df['B_EastGeomag']
#             else:
#                 df['%sSim' % QoIs] = df['B_NorthGeomag']
#             storm_Geospace = df[storm_start:storm_end].drop_duplicates()
#             if len(storm_Geospace) == 0:
#                 continue
#             else:
#                 # Deep GP model predictions: --------------------------------------------------------------------------
#                 file_path = station_path + station_to_evaluate_FFTfile[i]
#                 storm_X, storm_Y = test_general_matchGeospace(file_path, OMNI, QoIs, storm_Geospace.index)
#                 Joint_index = storm_Y.index.intersection(storm_Geospace.index)
#                 storm_X = storm_X.loc[Joint_index]
#                 storm_Y = storm_Y.loc[Joint_index]
#                 if not storm_X.shape[0] == 0:
#                     ## standardize data
#                     storm_Y = scipy.stats.yeojohnson(storm_Y.to_numpy().reshape(-1,1), lmbda=yeojohnson_lmbda)
#                     storm_X = scalerX.transform(storm_X.to_numpy())
#                     storm_Y = scalerY.transform(storm_Y)
#                     storm_X = torch.from_numpy(storm_X).type(torch.float32).to(get_device())
#                     storm_Y = torch.from_numpy(storm_Y).type(torch.float32).to(get_device())
#                     storm_Y = storm_Y.reshape(storm_Y.shape[0])
#                     storm_dataset = SuperMAGDataset_test(storm_X, storm_Y)
#                     storm_loader = DataLoader(storm_dataset, batch_size=4096)

#                     with torch.no_grad():
#                         pred_means, pred_var, lls = model.predict(storm_loader)
#                         sigma = sigma_GMM(pred_means, pred_var)
#                         qnorm_975 = scipy.stats.norm.ppf(0.975, loc=0, scale=1)
#                         pred_Y_DGP = pred_means.mean(0).cpu().numpy().reshape(-1,1)
#                         storm_Y_OG = scalerY.inverse_transform(storm_loader.dataset.Y.cpu().numpy().reshape(-1,1))
#                         pred_Y_OG = scalerY.inverse_transform(pred_Y_DGP)
#                         storm_Y_OG = np.array([invert_yeojhonson(x, yeojohnson_lmbda) for x in storm_Y_OG])
#                         pred_Y_OG = np.array([invert_yeojhonson(x, yeojohnson_lmbda) for x in pred_Y_OG])
#         #                 storm_Y_OG = scipy.special.inv_boxcox(storm_Y_OG, boxcox_lmbda, out=None)
#         #                 pred_Y_OG = scipy.special.inv_boxcox(pred_Y_OG, boxcox_lmbda, out=None)

#                         pred_Y_DGP_lr = pred_Y_DGP - qnorm_975*sigma.reshape(-1,1)
#                         pred_Y_DGP_up = pred_Y_DGP + qnorm_975*sigma.reshape(-1,1)
#                         pred_Y_OG_lr = np.array([invert_yeojhonson(x, yeojohnson_lmbda)
#                                                  for x in scalerY.inverse_transform(pred_Y_DGP_lr)]).flatten()
#                         pred_Y_OG_up = np.array([invert_yeojhonson(x, yeojohnson_lmbda)
#                                                  for x in scalerY.inverse_transform(pred_Y_DGP_up)]).flatten()

#         #                 pred_Y_OG_lr = scipy.special.inv_boxcox(scalerY.inverse_transform(pred_Y_DGP_lr),
#         #                                                         boxcox_lmbda, out=None).flatten()
#         #                 pred_Y_OG_up = scipy.special.inv_boxcox(scalerY.inverse_transform(pred_Y_DGP_up),
#         #                                                         boxcox_lmbda, out=None).flatten()
#                     pred_Y_OG_df = pd.DataFrame(pred_Y_OG)
#                     pred_Y_OG_df.index = Joint_index
#                     if QoIs == 'dBN':
#                         pred_Y_OG_df.columns = ['B_NorthGeomag']
#                     if QoIs == 'dBE':
#                         pred_Y_OG_df.columns = ['B_EastGeomag']
#                     if QoIs == 'dBH':
#                         pred_Y_OG_df.columns = ['dBH']
#                         storm_Y_OG_df = pd.DataFrame(storm_Y_OG)
#                         storm_Y_OG_df.index = Joint_index
#                         storm_Y_OG_df.columns = ['%s' % QoIs]
#                         storm_Y_OG_df.to_csv(path_prefix+"/Paper_deltaB/data/test_station/FFT/" + 
#                                             station_to_evaluate_FFTfile[i][0:3]+'_Obs%s.csv' % QoIs, index=False)
#                     pred_Y_OG_df['Year'] = [time.year for time in pred_Y_OG_df.index]
#                     pred_Y_OG_df['Month'] = [time.month for time in pred_Y_OG_df.index]
#                     pred_Y_OG_df['Day'] = [time.day for time in pred_Y_OG_df.index]
#                     pred_Y_OG_df['Hour'] = [time.hour for time in pred_Y_OG_df.index]
#                     pred_Y_OG_df['Min'] = [time.minute for time in pred_Y_OG_df.index]
#                     pred_Y_OG_df['Sec'] = [time.second for time in pred_Y_OG_df.index]
#                     pred_Y_OG_df.to_csv(path_prefix+"/Paper_deltaB/data/test_station/FFT/" + 
#                                         station_to_evaluate_FFTfile[i][0:3]+'_%s.csv' % QoIs, index=False)


# In[ ]:


if make_FFT_mergedfile:
    os.makedirs(path_prefix+"/Paper_deltaB/data/test_station/FFT_withUQ/merged/", exist_ok=True)
    for i in range(len(station_to_evaluate_FFT)):
        station_code = station_to_evaluate_FFT[i]
        ## merge files for FFT:
        dBE = pd.read_csv(path_prefix+"/Paper_deltaB/data/test_station/FFT_withUQ/QoIs/%s_dBE.csv" % station_code)
        dBH = pd.read_csv(path_prefix+"/Paper_deltaB/data/test_station/FFT_withUQ/QoIs/%s_dBH.csv" % station_code)
        dBN = pd.read_csv(path_prefix+"/Paper_deltaB/data/test_station/FFT_withUQ/QoIs/%s_dBN.csv" % station_code)
        station_HNE = pd.merge(pd.merge(dBH, dBN), dBE)
        station_HNE = station_HNE[['Year', 'Month', 'Day', 'Hour', 'Min', 'Sec',
                                   'dBH', 'B_EastGeomag', 'B_NorthGeomag', 'realization']]
        station_HNE.to_csv(path_prefix+'/Paper_deltaB/data/test_station/FFT_withUQ/merged/%s.csv' % station_code)


# In[ ]:


"""
FFT without uncertainty quantification
Deprecated: This chunk is outdated and will be removed in future versions.
"""
# if make_FFT_mergedfile:
#     for i in range(len(station_to_evaluate_FFT)):
#         station_code = station_to_evaluate_FFT[i]
#         ## merge files for FFT:
#         dBE = pd.read_csv(path_prefix+"/Paper_deltaB/data/test_station/FFT/%s_dBE.csv" % station_code)
#         dBH = pd.read_csv(path_prefix+"/Paper_deltaB/data/test_station/FFT/%s_dBH.csv" % station_code)
#         dBN = pd.read_csv(path_prefix+"/Paper_deltaB/data/test_station/FFT/%s_dBN.csv" % station_code)
#         station_HNE = pd.merge(pd.merge(dBH, dBN), dBE)
#         station_HNE = station_HNE[['Year', 'Month', 'Day', 'Hour', 'Min', 'Sec',
#                                    'dBH', 'B_EastGeomag', 'B_NorthGeomag']]
#         station_HNE.to_csv(path_prefix+'/Paper_deltaB/data/test_station/FFT/merged/%s.csv' % station_code)


# In[ ]:


# station_HNE = pd.read_csv('J:/Paper_deltaB/data/test_station/FFT/merged/ABK.csv')
# station_Obs = pd.read_csv('J:/Paper_deltaB/data/test_station/FFT/ABK_ObsdBH.csv')
# station_HNE['dBH_byEN'] = np.sqrt(station_HNE['B_EastGeomag'] **2 + station_HNE['B_NorthGeomag']**2)


# In[ ]:


# plt.rcParams['figure.figsize'] = [30,10]
# plt.plot(station_HNE['dBH_byEN'], color = 'red', label = 'FromNE', linewidth=1.5)
# plt.plot(station_HNE['dBH'], color = 'blue', label = 'direct', linewidth=1)
# # plt.plot(station_HNE['dBH'], color = 'blue', label = 'direct')
# plt.plot(station_Obs['dBH'], color = 'black', label = 'Obs', linewidth=1.5)
# MAE = mean_absolute_error(station_HNE['dBH_byEN'],station_Obs['dBH'])
# HSS = []
# threhold_list = [50, 200, 300, 400]
# for threshold in threhold_list:
#     HSS.append(compute_hss(station_Obs['dBH'].values,
#                            station_HNE['dBH_byEN'].values, threshold).iloc[0])
# plt.title('MAE: %d; HSS 50 nT: %.2f; HSS 200 nT: %.2f; HSS 300 nT: %.2f; HSS 400 nT: %.2f' % (MAE,
#                                                                                               HSS[0], HSS[1], HSS[2], HSS[3]),
#          fontsize=35)
# plt.xticks(size=30)
# plt.yticks(size=30)
# plt.legend(fontsize=30)
# plt.savefig('J:/Paper_deltaB/figure/TestSet3/NE_demo.png')


# In[ ]:


# plt.rcParams['figure.figsize'] = [15,7]
# # plt.plot(station_HNE['dBH_byEN'], color = 'red', label = 'FromNE')
# plt.plot(station_HNE['dBH'], color = 'red', label = 'direct')
# plt.plot(station_Obs['dBH'], color = 'black', label = 'Obs')
# MAE = mean_absolute_error(station_HNE['dBH'], station_Obs['dBH'])
# HSS = []
# threhold_list = [50, 200, 300, 400]
# for threshold in threhold_list:
#     HSS.append(compute_hss(station_Obs['dBH'].values,
#                            station_HNE['dBH'].values, threshold).iloc[0])
# plt.title('MAE: %d; HSS 50 nT: %.2f; HSS 200 nT: %.2f; HSS 300 nT: %.2f; HSS 400 nT: %.2f' % (MAE,
#                                                                                               HSS[0], HSS[1], HSS[2], HSS[3]))

