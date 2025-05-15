#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch.nn as nn
import pickle
from torch.utils.data import TensorDataset, DataLoader, Dataset
import re
from scipy.stats import percentileofscore
import astropy
import astropy.units as u
from astropy.coordinates import EarthLocation
from sunpy.coordinates import frames


# In[2]:


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
## percentile calculation
def get_score_MidLat(HSS_results):
    score_level = ["50nT", "100nT", "200nT", "300nT", "400nT"]
    level_name = score_level
    score_df = []
    for score in score_level:
        score_data = HSS_results[(np.abs(HSS_results["MagLat"]) < 50)][score].dropna()
        HSS_median = np.nanmedian(score_data)
        HSS_Q1 = np.nanquantile(score_data, 0.25)
        HSS_Q3 = np.nanquantile(score_data, 0.75)
        score_pd = pd.DataFrame([HSS_Q1, HSS_median, HSS_Q3]).T
        score_df.append(score_pd)
    score_df = pd.concat(score_df, axis=0)
    score_df.columns = ["Q1", "Median", "Q3"]
    score_df.index = level_name
    return score_df
def get_score_HighLat(HSS_results):
    score_level = ["50nT", "100nT", "200nT", "300nT", "400nT"]
    level_name = score_level
    score_df = []
    for score in score_level:
        score_data = HSS_results[(np.abs(HSS_results["MagLat"]) >= 50)][score].dropna()
        HSS_median = np.nanmedian(score_data)
        HSS_Q1 = np.nanquantile(score_data, 0.25)
        HSS_Q3 = np.nanquantile(score_data, 0.75)
        score_pd = pd.DataFrame([HSS_Q1, HSS_median, HSS_Q3]).T
        score_df.append(score_pd)
    score_df = pd.concat(score_df, axis=0)
    score_df.columns = ["Q1", "Median", "Q3"]
    score_df.index = level_name
    return score_df
def get_score_AuroralLat(HSS_results):
    score_level = ["50nT", "100nT", "200nT", "300nT", "400nT"]
    level_name = score_level
    score_df = []
    for score in score_level:
        score_data = HSS_results[(np.abs(HSS_results["MagLat"]) >= 60) & 
                                 (np.abs(HSS_results["MagLat"]) <= 70)][score].dropna()
        HSS_median = np.nanmedian(score_data)
        HSS_Q1 = np.nanquantile(score_data, 0.25)
        HSS_Q3 = np.nanquantile(score_data, 0.75)
        score_pd = pd.DataFrame([HSS_Q1, HSS_median, HSS_Q3]).T
        score_df.append(score_pd)
    score_df = pd.concat(score_df, axis=0)
    score_df.columns = ["Q1", "Median", "Q3"]
    score_df.index = level_name
    return score_df  
def get_score_AllLat(HSS_results):
    score_level = ["50nT", "100nT", "200nT", "300nT", "400nT"]
    level_name = score_level
    score_df = []
    for score in score_level:
        score_data = HSS_results[score].dropna()
        HSS_median = np.nanmedian(score_data)
        HSS_Q1 = np.nanquantile(score_data, 0.25)
        HSS_Q3 = np.nanquantile(score_data, 0.75)
        score_pd = pd.DataFrame([HSS_Q1, HSS_median, HSS_Q3]).T
        score_df.append(score_pd)
    score_df = pd.concat(score_df, axis=0)
    score_df.columns = ["Q1", "Median", "Q3"]
    score_df.index = level_name
    return score_df


# In[3]:


class SuperMAGDataset(Dataset):
    def __init__(self, X_path, Y_path):
        self.X = torch.load(X_path)
        self.Y = torch.load(Y_path)
    
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        return {'X': self.X[idx], "Y": self.Y[idx]}


# In[4]:


## 5 minutes window median.
path_prefix = "J:"
path = path_prefix + "/Paper_deltaB/data/train/"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# station_path = path + 'data/QoIs_SM_20min_SM_lon_lat/'
# N_WORKERS = 10
# assert N_WORKERS<= os.cpu_count()
QoIs = "dBE"
yeojohnson_lmbda = 1
window_max = 10
multihour_ahead = True
hour = 1
WithDst=True
percentile_estimate = False
UQ=True
make_testset1 = False
# maximum=False
if WithDst:
    with open(path + "scaler/scalerX_%s.pkl" % window_max,'rb') as f:
        scalerX = pickle.load(f)
    with open(path + "scaler/scalerY_%s_%s.pkl" % (QoIs, window_max),'rb') as f:
        scalerY = pickle.load(f)
    X_shape = 96
else:
    QoIs = "dBH"
    multihour_ahead = False
    with open(path + "scaler/scalerX_%s_noDst.pkl" % window_max,'rb') as f:
        scalerX = pickle.load(f)
    with open(path + "scaler/scalerY_%s_%s_noDst.pkl" % (QoIs, window_max),'rb') as f:
        scalerY = pickle.load(f)
    X_shape = 83
# val_dataset = SuperMAGDataset(res_folder + 'val_X_%s.pt' % QoIs, res_folder + 'val_Y_%s.pt' % QoIs)
# val_loader = DataLoader(val_dataset,
#                         batch_size=1024)
# X_shape = val_dataset.X.shape[1]
torch.manual_seed(20241023)


# In[5]:


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


# In[6]:


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


# In[7]:


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


# In[8]:


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
                lls.append(self.likelihood.log_marginal(y_batch, self(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)


# In[9]:


if WithDst:
    OMNI = pd.read_pickle(path + "../Input/OMNI_paper_5m_feature.pkl").dropna()
else:
    OMNI = pd.read_pickle(path + "../Input/OMNI_paper_5m_feature.pkl").dropna()
    OMNI.drop('Dst', axis=1, inplace=True)
    OMNI.drop(columns=OMNI.columns[80:93], axis=1, inplace=True)
deltaT_SW = datetime.timedelta(minutes=45)
if multihour_ahead:
    model = DeepGaussianProcess(X_shape)
    model_ahead = DeepGaussianProcess(X_shape)
    deltaT = datetime.timedelta(hours=hour)
    OMNI.index = OMNI.index + deltaT
    state_dict = torch.load(path + "model/paper/model_state_%s_10T.pth" % QoIs)
    state_dict_ahead = torch.load(path + "model/paper/model_state_%s_10T_%sh_ahead.pth" % (QoIs, hour))
    model.load_state_dict(state_dict)
    model_ahead.load_state_dict(state_dict_ahead)
    if torch.cuda.is_available():
        model = model.cuda()
        model_ahead = model_ahead.cuda()
else:
    if WithDst:
        state_dict = torch.load(path + "model/paper/model_state_%s_10T.pth" % QoIs)
    else:
        state_dict = torch.load(path + "model/paper/model_state_%s_10T_noDst.pth" % QoIs)
    model = DeepGaussianProcess(X_shape)
    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model = model.cuda()


# In[10]:


class SuperMAGDataset_test(Dataset):
    def __init__(self, storm_X_tensor, storm_Y_tensor):
        self.X = storm_X_tensor
        self.Y = storm_Y_tensor
    
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        return {'X': self.X[idx], "Y": self.Y[idx]}


# In[11]:


year = 2015
station_path = path_prefix+"/Paper_deltaB/data/AllStations_AllYear_1min_raw/"
station_file = sorted([x for x in os.listdir(station_path) if bool(re.match(r'.*_%s\.pkl$' % year, x))])
Geospace_path = path_prefix+"/Paper_deltaB/data/Geospace_Qusai/"
Geospace_stations= sorted([x.split('.csv')[0] for x in os.listdir(Geospace_path)])
My_stations = sorted([x.split('_2015')[0] for x in station_file])
station_to_evaluate = sorted(list(set(Geospace_stations) & set(My_stations)))
station_to_evaluate_file = sorted([x for x in station_file if (x.split('_2015')[0] in station_to_evaluate)])


# In[12]:


test_storm_date_list = [datetime.datetime(2015, 2, 16, 19, 24),
                        datetime.datetime(2015, 3, 17, 4, 7),
                       datetime.datetime(2015, 4, 9, 21, 52),
                       datetime.datetime(2015, 4, 14, 12, 55),
                       datetime.datetime(2015, 5, 12, 18, 5),
                       datetime.datetime(2015, 5, 18, 10, 12),
                       datetime.datetime(2015, 6, 7, 10, 30),
                       datetime.datetime(2015, 6, 22, 5, 0),
                       datetime.datetime(2015, 7, 4, 13, 6),
                       datetime.datetime(2015, 7, 10, 22, 21),
                       datetime.datetime(2015, 7, 23, 1, 51),
                       datetime.datetime(2015, 8, 15, 8, 4),
                       datetime.datetime(2015, 8, 26, 5, 45),
                       datetime.datetime(2015, 9, 7, 13, 13),
                       datetime.datetime(2015, 9, 8, 21, 45),
                       datetime.datetime(2015, 9, 20, 5, 46),
                       datetime.datetime(2015, 10, 4, 0, 30),
                       datetime.datetime(2015, 10, 7, 1, 41),
                       datetime.datetime(2015, 11, 3, 5, 31),
                       datetime.datetime(2015, 11, 6, 18, 9),
                       datetime.datetime(2015, 11, 30, 6, 9),
                       datetime.datetime(2015, 12, 19, 16, 13)]
upper = 48
lower = 6


# In[13]:


def DGP_pred(OMNI, Joint_index, station_data, QoIs, model_in):
    test_OMNI_X = OMNI.loc[Joint_index]
    test_SuperMAG = station_data.loc[Joint_index]
    storm_Y = test_SuperMAG[QoIs]
    storm_X = pd.concat([test_OMNI_X,
                        test_SuperMAG[["SM_lat"]],
                        np.cos(test_SuperMAG["SM_lon"].astype(np.float32) / 180 * np.pi),
                        np.sin(test_SuperMAG["SM_lon"].astype(np.float32) / 180 * np.pi)
                       ],
                       axis = 1)
    storm_X.columns = storm_X.columns[:-2].to_list() + ["CosLon", "SinLon"]
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
        sigma = sigma_GMM(pred_means, pred_var)
        qnorm_975 = scipy.stats.norm.ppf(0.975, loc=0, scale=1)
        pred_Y_DGP = pred_means.mean(0).cpu().numpy().reshape(-1,1)
        storm_Y_OG = scalerY.inverse_transform(storm_loader.dataset.Y.cpu().numpy().reshape(-1,1))
        pred_Y_OG = scalerY.inverse_transform(pred_Y_DGP)
        storm_Y_OG = np.array([invert_yeojhonson(x, yeojohnson_lmbda) for x in storm_Y_OG])
        pred_Y_OG = np.array([invert_yeojhonson(x, yeojohnson_lmbda) for x in pred_Y_OG])
    return storm_Y_OG, pred_Y_OG


# In[14]:


if percentile_estimate:
    ## estimate the percentile
    thresholds = np.array([50, 100, 200, 300, 400])

    station_p_df = []
    for i in track(range(len(station_to_evaluate_file))):
        file_path = station_path + station_to_evaluate_file[i]
        station_code = station_to_evaluate_file[i].split('_2015.pkl')[0]
        station_data = pd.read_pickle(file_path)
        station_data = station_data.dropna()
        station_data.index = station_data['Time']
        station_storms = []
        for j in range(len(test_storm_date_list)):
            storm_onset = test_storm_date_list[j]
            storm_start = storm_onset - datetime.timedelta(hours = lower)
            storm_end = storm_onset + datetime.timedelta(hours = upper)
            station_storms.append(station_data['dBH'][storm_start:storm_end])
        station_storms = pd.concat(station_storms, axis=0)
        percentile = percentileofscore(station_storms, thresholds, kind='rank')
        percentile_df = pd.DataFrame([{
            '50nT': percentile[0],
            '100nT': percentile[1],
            '200nT': percentile[2],
            '300nT': percentile[3],
            '400nT': percentile[4],
            'Station': station_code
        }])
        station_p_df.append(percentile_df)
    station_p_df = pd.concat(station_p_df, axis=0)
    ## calculate the geomagnetic latitude and longitude.
    station_info = pd.read_csv(path_prefix + "/Paper_deltaB/data/Input/station_info.csv")
    station_location = station_info.iloc[:,0:3]
    station_location.columns = ["Station", "GEOLON", "GEOLAT"]
    station_p_loc = pd.merge(station_location, station_p_df, on='Station', how='inner')
    obs_time = datetime.datetime(2015, 6, 1)
    Earth_loc = astropy.coordinates.EarthLocation(lat=station_p_loc["GEOLAT"].values*u.deg,
                                         lon=station_p_loc["GEOLON"].values*u.deg)
    Geographic = astropy.coordinates.ITRS(x=Earth_loc.x, y=Earth_loc.y, z=Earth_loc.z, obstime=obs_time)
    target_coord = frames.Geomagnetic(magnetic_model='igrf13', obstime=obs_time)
    GeoMagnetic = Geographic.transform_to(target_coord)
    GeoMAG = pd.concat([pd.Series(GeoMagnetic.lon.value),
                        pd.Series(GeoMagnetic.lat.value)],
                       axis = 1)
    GeoMAG.columns = ["MagLon", "MagLat"]
    station_p_loc = pd.concat([station_p_loc, GeoMAG], axis=1)
    p_tab = pd.concat([get_score_MidLat(station_p_loc)['Median'].to_frame().T,
           get_score_HighLat(station_p_loc)['Median'].to_frame().T,
           get_score_AuroralLat(station_p_loc)['Median'].to_frame().T,
           get_score_AllLat(station_p_loc)['Median'].to_frame().T],
          axis=0)
    p_tab.index = ['Mid', 'High', 'Auroral', 'All']
    print(p_tab.round(0))


# In[15]:


def DGP_pred_withUQ(OMNI, Joint_index, station_data, QoIs, model_in):
    test_OMNI_X = OMNI.loc[Joint_index]
    test_SuperMAG = station_data.loc[Joint_index]
    storm_Y = test_SuperMAG[QoIs]
    storm_X = pd.concat([test_OMNI_X,
                        test_SuperMAG[["SM_lat"]],
                        np.cos(test_SuperMAG["SM_lon"].astype(np.float32) / 180 * np.pi),
                        np.sin(test_SuperMAG["SM_lon"].astype(np.float32) / 180 * np.pi)
                       ],
                       axis = 1)
    storm_X.columns = storm_X.columns[:-2].to_list() + ["CosLon", "SinLon"]
    if not storm_X.shape[0] == 0:
        ## standardize data
        storm_X = scalerX.transform(storm_X.to_numpy())
        storm_X = torch.from_numpy(storm_X).type(torch.float32).to(get_device())
        storm_Y = scalerY.transform(storm_Y.to_numpy().reshape(-1,1))
        storm_Y = torch.from_numpy(storm_Y).type(torch.float32).to(get_device())
        storm_Y = storm_Y.reshape(storm_Y.shape[0])
        storm_dataset = SuperMAGDataset_test(storm_X, storm_Y)
        storm_loader = DataLoader(storm_dataset, batch_size=4096)
    with torch.no_grad():
        pred_means, pred_var, lls = model_in.predict(storm_loader)
        sigma = sigma_GMM(pred_means, pred_var)
        qnorm_975 = scipy.stats.norm.ppf(0.975, loc=0, scale=1)
        pred_Y_DGP = pred_means.mean(0).cpu().numpy().reshape(-1,1)
        storm_Y_OG = scalerY.inverse_transform(storm_loader.dataset.Y.cpu().numpy().reshape(-1,1))
        pred_Y_OG = scalerY.inverse_transform(pred_Y_DGP)
        pred_Y_DGP_lr = pred_Y_DGP - qnorm_975*sigma.reshape(-1,1)
        pred_Y_DGP_up = pred_Y_DGP + qnorm_975*sigma.reshape(-1,1)
        pred_Y_OG_lr = scalerY.inverse_transform(pred_Y_DGP_lr)
        pred_Y_OG_up = scalerY.inverse_transform(pred_Y_DGP_up)
    return storm_Y_OG, pred_Y_OG, pred_Y_OG_lr, pred_Y_OG_up


# In[16]:


# if UQ:
#     score_df = []
#     for i in track(range(len(station_to_evaluate_file))):
#         # Deep GP model predictions: --------------------------------------------------------------------------
#         file_path = station_path + station_to_evaluate_file[i]
#         station_data = pd.read_pickle(file_path)
#         station_data = station_data.dropna()
#         station_data.index = station_data['Time']
# #         if maximum:
# #             station_max = station_data[QoIs].resample('10T', label='left').apply(lambda x : max(x,
# # key = abs, default = np.NaN))
# #             station_data = pd.DataFrame(station_max).join(station_data[["SM_lon", "SM_lat"]], how='left').dropna()
# #         else:
# #             station_data = station_data[[QoIs, "SM_lon", "SM_lat"]]
#         station_data = station_data[[QoIs, "SM_lon", "SM_lat"]]
#         station_storm_df = []
#         for j in range(len(test_storm_date_list)):
#             storm_onset = test_storm_date_list[j]
#             storm_start = storm_onset - datetime.timedelta(hours = lower)
#             storm_end = storm_onset + datetime.timedelta(hours = upper)
#             station_storm_df.append(station_data.loc[storm_start:storm_end].drop_duplicates())
#         station_storm_df = pd.concat(station_storm_df, axis=0)
#         Joint_index = OMNI.index.intersection(station_storm_df.index)
#         #         if multihour_ahead:
#         #             ## further consider the data availability of observation persitence
#         #             Joint_index = station_data.index.intersection(Joint_index-deltaT-deltaT_SW)+deltaT+deltaT_SW
#         #         else:
#         #             Joint_index = station_data.index.intersection(Joint_index-deltaT_SW)+deltaT_SW
#             ## deltaT_SW (this one becomes the model persistence when multihour_ahead = True)
#         if len(station_storm_df) != 0:
#             if multihour_ahead:
#                 storm_Y_OG, pred_Y_OG, pred_Y_OG_lr, pred_Y_OG_up = DGP_pred_withUQ(OMNI, Joint_index,
#                                                                              station_storm_df, QoIs, model_ahead)
#             else:
#                 storm_Y_OG, pred_Y_OG, pred_Y_OG_lr, pred_Y_OG_up = DGP_pred_withUQ(OMNI, Joint_index,
#                                                                                     station_storm_df, QoIs, model)
#             pred_Y_OG_lr = np.clip(pred_Y_OG_lr, 0, np.inf)
#             alpha = 0.05
#             ## x < l:
#             indicator_x_less_l = np.less(pred_Y_OG, pred_Y_OG_lr).astype(int)
#             indicator_u_less_x = np.less(pred_Y_OG_up, pred_Y_OG).astype(int)
#             term1 = (pred_Y_OG_up - pred_Y_OG_lr) 
#             term2 = 2/alpha*(pred_Y_OG_lr - pred_Y_OG) * indicator_x_less_l
#             term3 = 2/alpha*(pred_Y_OG - pred_Y_OG_up)*indicator_u_less_x
#             interval_score = np.mean(term1+term2+term3)
#             score_df_station = pd.DataFrame([{'Station': station_to_evaluate_file[i].split('_')[0],
#                          'interval_score': interval_score}])
#             score_df.append(score_df_station)
#     score_df = pd.concat(score_df, axis=0)
#     if multihour_ahead:
#         res_path = path_prefix+"/Paper_deltaB/data/test_station/TestSet1/%sT_%sh_ahead/%s/" % (window_max, hour, QoIs)
#     else:
#         res_path = path_prefix+"/Paper_deltaB/data/test_station/TestSet1/%sT/%s/" % (window_max, QoIs)
# #     if maximum:
# #         score_df.to_csv(res_path+"interval_score_max.csv", index=False)
# #     else:
# #         score_df.to_csv(res_path+"interval_score.csv", index=False)
#     score_df.to_csv(res_path+"interval_score.csv", index=False)


# In[17]:


if UQ:
    score_df = []
    coverage_df = []
    width_df = []
    for i in track(range(len(station_to_evaluate_file))):
        # Deep GP model predictions: --------------------------------------------------------------------------
        file_path = station_path + station_to_evaluate_file[i]
        station_data = pd.read_pickle(file_path)
        station_data = station_data.dropna()
        station_data.index = station_data['Time']
#         if maximum:
#             station_max = station_data[QoIs].resample('10T', label='left').apply(lambda x : max(x,
# key = abs, default = np.NaN))
#             station_data = pd.DataFrame(station_max).join(station_data[["SM_lon", "SM_lat"]], how='left').dropna()
#         else:
#             station_data = station_data[[QoIs, "SM_lon", "SM_lat"]]
        station_data = station_data[[QoIs, "SM_lon", "SM_lat"]]
        station_storm_df = []
        for j in range(len(test_storm_date_list)):
            storm_onset = test_storm_date_list[j]
            storm_start = storm_onset - datetime.timedelta(hours = lower)
            storm_end = storm_onset + datetime.timedelta(hours = upper)
            station_storm_df.append(station_data.loc[storm_start:storm_end].drop_duplicates())
        station_storm_df = pd.concat(station_storm_df, axis=0)
        Joint_index = OMNI.index.intersection(station_storm_df.index)
        #         if multihour_ahead:
        #             ## further consider the data availability of observation persitence
        #             Joint_index = station_data.index.intersection(Joint_index-deltaT-deltaT_SW)+deltaT+deltaT_SW
        #         else:
        #             Joint_index = station_data.index.intersection(Joint_index-deltaT_SW)+deltaT_SW
            ## deltaT_SW (this one becomes the model persistence when multihour_ahead = True)
        if len(station_storm_df) != 0:
            if multihour_ahead:
                storm_Y_OG, pred_Y_OG, pred_Y_OG_lr, pred_Y_OG_up = DGP_pred_withUQ(OMNI, Joint_index,
                                                                             station_storm_df, QoIs, model_ahead)
            else:
                storm_Y_OG, pred_Y_OG, pred_Y_OG_lr, pred_Y_OG_up = DGP_pred_withUQ(OMNI, Joint_index,
                                                                                    station_storm_df, QoIs, model)
            if QoIs == 'dBH':
                pred_Y_OG_lr = np.clip(pred_Y_OG_lr, 0, np.inf)
            ## interval score:
            alpha = 0.05
            ## x < l:
            indicator_x_less_l = np.less(storm_Y_OG, pred_Y_OG_lr).astype(int)
            indicator_u_less_x = np.less(pred_Y_OG_up, storm_Y_OG).astype(int)
            term1 = (pred_Y_OG_up - pred_Y_OG_lr) 
            term2 = 2/alpha*(pred_Y_OG_lr - storm_Y_OG)*indicator_x_less_l
            term3 = 2/alpha*(storm_Y_OG - pred_Y_OG_up)*indicator_u_less_x
            interval_score = np.mean(term1+term2+term3)
            score_df_station = pd.DataFrame([{'Station': station_to_evaluate_file[i].split('_')[0],
                         'interval_score': interval_score}])
            score_df.append(score_df_station)
            ## interval width:
            interval_width = np.mean(term1)
            width_df_station = pd.DataFrame([{'Station': station_to_evaluate_file[i].split('_')[0],
             'interval_width': interval_width}])
            width_df.append(width_df_station)
            ## coverage rate:
            upper_cover = (storm_Y_OG < pred_Y_OG_up)
            lower_cover = (storm_Y_OG > pred_Y_OG_lr)
            cover_rate = np.mean(np.logical_and(lower_cover, upper_cover))
            coverage_df_station = pd.DataFrame([{'Station': station_to_evaluate_file[i].split('_')[0],
                         'coverage': cover_rate}])
            coverage_df.append(coverage_df_station)
    score_df = pd.concat(score_df, axis=0)
    coverage_df = pd.concat(coverage_df, axis=0)
    width_df = pd.concat(width_df, axis=0)
    if multihour_ahead:
        res_path = path_prefix+"/Paper_deltaB/data/test_station/TestSet1/%sT_%sh_ahead/%s/" % (window_max, hour, QoIs)
    else:
        res_path = path_prefix+"/Paper_deltaB/data/test_station/TestSet1/%sT/%s/" % (window_max, QoIs)
#     if maximum:
#         score_df.to_csv(res_path+"interval_score_max.csv", index=False)
#     else:
#         score_df.to_csv(res_path+"interval_score.csv", index=False)
    score_df.to_csv(res_path+"interval_score.csv", index=False)
    coverage_df.to_csv(res_path+"coverage.csv", index=False)
    width_df.to_csv(res_path+"interval_width.csv", index=False)


# In[18]:


# if UQ:
#     coverage_df = []
#     for i in track(range(len(station_to_evaluate_file))):
#         # Deep GP model predictions: --------------------------------------------------------------------------
#         file_path = station_path + station_to_evaluate_file[i]
#         station_data = pd.read_pickle(file_path)
#         station_data = station_data.dropna()
#         station_data.index = station_data['Time']
# #         if maximum:
# #             station_max = station_data[QoIs].resample('10T', label='left').apply(lambda x : max(x, key = abs, default = np.NaN))
# #             station_data = pd.DataFrame(station_max).join(station_data[["SM_lon", "SM_lat"]], how='left').dropna()
# #         else:
# #             station_data = station_data[[QoIs, "SM_lon", "SM_lat"]]
#         station_data = station_data[[QoIs, "SM_lon", "SM_lat"]]
#         station_storm_df = []
#         for j in range(len(test_storm_date_list)):
#             storm_onset = test_storm_date_list[j]
#             storm_start = storm_onset - datetime.timedelta(hours = lower)
#             storm_end = storm_onset + datetime.timedelta(hours = upper)
#             station_storm_df.append(station_data.loc[storm_start:storm_end].drop_duplicates())
#         station_storm_df = pd.concat(station_storm_df, axis=0)
#         Joint_index = OMNI.index.intersection(station_storm_df.index)
#         #         if multihour_ahead:
#         #             ## further consider the data availability of observation persitence
#         #             Joint_index = station_data.index.intersection(Joint_index-deltaT-deltaT_SW)+deltaT+deltaT_SW
#         #         else:
#         #             Joint_index = station_data.index.intersection(Joint_index-deltaT_SW)+deltaT_SW
#             ## deltaT_SW (this one becomes the model persistence when multihour_ahead = True)
#         if len(station_storm_df) != 0:
#             if multihour_ahead:
#                 storm_Y_OG, pred_Y_OG, pred_Y_OG_lr, pred_Y_OG_up = DGP_pred_withUQ(OMNI, Joint_index,
#                                                                              station_data, QoIs, model_ahead)
#             else:
#                 storm_Y_OG, pred_Y_OG, pred_Y_OG_lr, pred_Y_OG_up = DGP_pred_withUQ(OMNI, Joint_index,
#                                                                                     station_storm_df, QoIs, model)
#             upper_cover = (storm_Y_OG < pred_Y_OG_up)
#             lower_cover = (storm_Y_OG > pred_Y_OG_lr)
#             cover_rate = np.mean(np.logical_and(lower_cover, upper_cover))
#             coverage_df_station = pd.DataFrame([{'Station': station_to_evaluate_file[i].split('_')[0],
#                          'coverage': cover_rate}])
#             coverage_df.append(coverage_df_station)
#     coverage_df = pd.concat(coverage_df, axis=0)
#     if multihour_ahead:
#         res_path = path_prefix+"/Paper_deltaB/data/test_station/TestSet1/%sT_%sh_ahead/%s/" % (window_max, hour, QoIs)
#     else:
#         res_path = path_prefix+"/Paper_deltaB/data/test_station/TestSet1/%sT/%s/" % (window_max, QoIs)
#     coverage_df.to_csv(res_path+"coverage.csv", index=False)
# #     if maximum:
# #         coverage_df.to_csv(res_path+"coverage_max.csv", index=False)
# #     else:
# #         coverage_df.to_csv(res_path+"coverage.csv", index=False)


# In[19]:


if make_testset1:
    HSS_df = []
    for i in track(range(len(station_to_evaluate_file))):
    #     Geospace model predictions: --------------------------------------------------------------------------
        GeospaceSim_file = Geospace_path + station_to_evaluate_file[i].split('_2015.pkl')[0] + '.csv'
        df = pd.read_csv(GeospaceSim_file)
        ## Apply the function to each row in the DataFrame
        df['Datetime'] = df.apply(convert_to_datetime, axis=1)
        df.set_index('Datetime', inplace=True)
        if QoIs == "dBH":
            df['%sSim' % QoIs] = np.sqrt(df['BeSim']**2 + df['BnSim']**2)
            df['%sObs' % QoIs] = np.sqrt(df['BeObs']**2 + df['BnObs']**2)
        elif QoIs == "dBE":
            df['%sSim' % QoIs] = df['BeSim']
            df['%sObs' % QoIs] = df['BeObs']
        else:
            df['%sSim' % QoIs] = df['BnSim']
            df['%sObs' % QoIs] = df['BnObs']
        Geospace_station_i_allstorms = []
        for j in range(len(test_storm_date_list)):
            storm_onset = test_storm_date_list[j]
            storm_start = storm_onset - datetime.timedelta(hours = lower)
            storm_end = storm_onset + datetime.timedelta(hours = upper)
            Geospace_station_i_allstorms.append(df[storm_start:storm_end].drop_duplicates())
        Geospace_station_i_allstorms = pd.concat(Geospace_station_i_allstorms, axis=0)
        if len(Geospace_station_i_allstorms) == 0:
            continue
        else:
            # Deep GP model predictions: --------------------------------------------------------------------------
            file_path = station_path + station_to_evaluate_file[i]
            station_data = pd.read_pickle(file_path)
            station_data = station_data.dropna()
            station_data.index = station_data['Time']
            station_data = station_data[[QoIs, "SM_lon", "SM_lat"]]
            OMNI_index = OMNI.index.intersection(Geospace_station_i_allstorms.index)
            Joint_index = OMNI_index.intersection(station_data.index)
            if multihour_ahead:
                ## further consider the data availability of observation persitence
                Joint_index = station_data.index.intersection(Joint_index-deltaT-deltaT_SW)+deltaT+deltaT_SW
            else:
                Joint_index = station_data.index.intersection(Joint_index-deltaT_SW)+deltaT_SW
            Geospace_station_i_allstorms = Geospace_station_i_allstorms.loc[Joint_index].drop_duplicates()
            ## deltaT_SW (this one becomes the model persistence when multihour_ahead = True)
            storm_Y_OG, pred_Y_OG = DGP_pred(OMNI, Joint_index, station_data, QoIs, model)
            ## the deltaT_SW + hour ahead prediction
            if multihour_ahead:
                storm_Y_OG_ahead, pred_Y_OG_ahead = DGP_pred(OMNI, Joint_index, station_data, QoIs, model_ahead)
                ## the observation persistence
                observation_persistence = station_data.loc[Joint_index-deltaT-deltaT_SW][QoIs].to_numpy().reshape(-1,1)
            else:
                observation_persistence = station_data.loc[Joint_index-deltaT_SW][QoIs].to_numpy().reshape(-1,1)
        ## evaluation metrics: --------------------------------------------------------------------------------
            threholds = [50, 100, 200, 300, 400]
            colnames = ["HSS_L1", "HSS_L2", "HSS_L3", "HSS_L4", "HSS_L5", "Station", "MAE", "SignRate", "Method"]
            ## DGP HSS
            HSS = []
            for threshold in threholds:
                HSS.append(compute_hss(np.sign(storm_Y_OG) * storm_Y_OG,
                                       np.sign(storm_Y_OG) * pred_Y_OG, threshold).iloc[0])
            HSS_pd = pd.DataFrame(HSS).T
            HSS_pd["Station"] = station_to_evaluate_file[i][0:3]
            MAE = mean_absolute_error(storm_Y_OG, pred_Y_OG)
            HSS_pd["MAE"] = int(MAE)
            HSS_pd["SignRate"] = np.mean(np.sign(storm_Y_OG) == np.sign(pred_Y_OG))
            if multihour_ahead:
                HSS_pd["Method"] = "GeoDGP_persistence"
            else:
                HSS_pd["Method"] = "GeoDGP"
            HSS_pd.columns = colnames
            HSS_df.append(HSS_pd)
            ## Geospace HSS calculation
            HSS = []
            Geospace_Y_OG = Geospace_station_i_allstorms['%sSim' % QoIs].to_numpy().reshape(-1,1)
            for threshold in threholds:
                HSS.append(compute_hss(np.sign(storm_Y_OG) * storm_Y_OG,
                                       np.sign(storm_Y_OG) * Geospace_Y_OG,
                                       threshold).iloc[0])
            HSS_pd = pd.DataFrame(HSS).T
            HSS_pd["Station"] = station_to_evaluate_file[i][0:3]
            MAE = mean_absolute_error(storm_Y_OG, Geospace_Y_OG)
            HSS_pd["MAE"] = int(MAE)
            HSS_pd["SignRate"] = np.mean(np.sign(storm_Y_OG) ==  np.sign(Geospace_Y_OG))
            HSS_pd["Method"] = "Geospace"
            HSS_pd.columns = colnames
            HSS_df.append(HSS_pd)
            if multihour_ahead:
                ## DGP model Persistence HSS calculation
                HSS = []
                for threshold in threholds:
                    HSS.append(compute_hss(np.sign(storm_Y_OG) * storm_Y_OG,
                                           np.sign(storm_Y_OG) * pred_Y_OG_ahead, threshold).iloc[0])
                HSS_pd = pd.DataFrame(HSS).T
                HSS_pd["Station"] = station_to_evaluate_file[i][0:3]
                MAE = mean_absolute_error(storm_Y_OG, pred_Y_OG_ahead)
                HSS_pd["MAE"] = int(MAE)
                HSS_pd["SignRate"] = np.mean(np.sign(storm_Y_OG) == np.sign(pred_Y_OG_ahead))
                HSS_pd["Method"] = "GeoDGP_Ahead"
                HSS_pd.columns = colnames
                HSS_df.append(HSS_pd)
                ## Observation Persistence HSS calculation
                HSS = []
                for threshold in threholds:
                    HSS.append(compute_hss(np.sign(storm_Y_OG) * storm_Y_OG,
                                           np.sign(storm_Y_OG) * observation_persistence, threshold).iloc[0])
                HSS_pd = pd.DataFrame(HSS).T
                HSS_pd["Station"] = station_to_evaluate_file[i][0:3]
                MAE = mean_absolute_error(storm_Y_OG, observation_persistence)
                HSS_pd["MAE"] = int(MAE)
                HSS_pd["SignRate"] = np.mean(np.sign(storm_Y_OG) == np.sign(observation_persistence))
                HSS_pd["Method"] = "Observation_persistence"
                HSS_pd.columns = colnames
                HSS_df.append(HSS_pd)
            else:
                ## Observation Persistence HSS calculation
                HSS = []
                for threshold in threholds:
                    HSS.append(compute_hss(np.sign(storm_Y_OG) * storm_Y_OG,
                                           np.sign(storm_Y_OG) * observation_persistence, threshold).iloc[0])
                HSS_pd = pd.DataFrame(HSS).T
                HSS_pd["Station"] = station_to_evaluate_file[i][0:3]
                MAE = mean_absolute_error(storm_Y_OG, observation_persistence)
                HSS_pd["MAE"] = int(MAE)
                HSS_pd["SignRate"] = np.mean(np.sign(storm_Y_OG) == np.sign(observation_persistence))
                HSS_pd["Method"] = "Observation_persistence"
                HSS_pd.columns = colnames
                HSS_df.append(HSS_pd)
    HSS_df = pd.concat(HSS_df, axis=0)
    ## set results path
    if multihour_ahead:
        res_path = path_prefix+"/Paper_deltaB/data/test_station/TestSet1/%sT_%sh_ahead/%s/" % (window_max, hour, QoIs)
    else:
        if WithDst:
            res_path = path_prefix+"/Paper_deltaB/data/test_station/TestSet1/%sT/%s/" % (window_max, QoIs)
        else:
            res_path = path_prefix+"/Paper_deltaB/data/test_station/TestSet1/NoDst/%sT/%s/" % (window_max, QoIs)
    # res_path = "J:/Paper_deltaB/data/test_station/TestSet1/%sT/%s/" % (window_max, QoIs)
    os.makedirs(res_path, exist_ok = True)
    HSS_df.to_csv(res_path+"HSS.csv", index=False)

