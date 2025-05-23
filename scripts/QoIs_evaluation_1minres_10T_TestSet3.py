import torch
import gpytorch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import scipy
from rich.progress import track
import datetime
from sklearn.metrics import mean_absolute_error
import pickle
from torch.utils.data import DataLoader, Dataset
import matplotlib.dates as mdates
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from src.utils import *
from src.model import *

# In[ ]:
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
            ## prediction interval
            sigma = sigma_GMM(pred_means, pred_var)
            qnorm_975 = scipy.stats.norm.ppf(0.975, loc=0, scale=1)
            pred_Y_DGP_lr = pred_Y_DGP - qnorm_975*sigma.reshape(-1,1)
            pred_Y_DGP_up = pred_Y_DGP + qnorm_975*sigma.reshape(-1,1)
            pred_Y_OG_lr = scalerY.inverse_transform(pred_Y_DGP_lr).flatten()
            pred_Y_OG_up = scalerY.inverse_transform(pred_Y_DGP_up).flatten()
    return storm_Y_OG, pred_Y_OG, pred_Y_OG_lr, pred_Y_OG_up

# In[ ]:
path = root_path+"/data/train/"
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
deltaT_SW = datetime.timedelta(minutes=45)
if multihour_ahead:
    OMNI = pd.read_pickle(path + "../Input/OMNI_May10_5m_feature_ACE.pkl").dropna()
    deltaT = datetime.timedelta(hours=1)
    OMNI.index = OMNI.index +deltaT
    res_path = root_path+"/data/test_station/TestSet3_IMF_ACE/%sT_1h_ahead/%s/" % (window_max, QoIs)
else:
    OMNI = pd.read_pickle(path + "../Input/OMNI_May10_5m_feature_ACE.pkl").dropna()   
    res_path = root_path+"/data/test_station/TestSet3_IMF_ACE/%sT/%s/" % (window_max, QoIs)

os.makedirs(res_path, exist_ok = True)
make_FFTfile = False
make_TestSet3 = True
make_FFT_mergedfile = False
make_TestSet3_figure = False
station_to_evaluate_FFT = ['ABK', 'OTT', 'FRD']
station_to_visualize = sorted(['ABK', 'NEW', 'OTT', 'MEA', 'WNG', 'YKC'])
torch.manual_seed(20241030)

if multihour_ahead:
    # 1-hour model
    model = DeepGaussianProcess(X_shape,
                                DEVICE,
                 num_hidden1_dims = 20,
                 num_hidden2_dims = 10,
                 num_hidden3_dims = 10)
    state_dict = torch.load(path + "model/paper/model_state_%s_10T_1h_ahead.pth" % QoIs)
    model.load_state_dict(state_dict)
    # model persistence
    model_p = DeepGaussianProcess(X_shape,
                                DEVICE,
                 num_hidden1_dims = 20,
                 num_hidden2_dims = 10,
                 num_hidden3_dims = 10)
    state_dict = torch.load(path + "model/paper/model_state_%s_10T.pth" % QoIs)
    model_p.load_state_dict(state_dict)
    ## load to GPU
    if torch.cuda.is_available():
        model = model.cuda()
        model_p = model_p.cuda()
else:
    model = DeepGaussianProcess(X_shape,
                                DEVICE,
                 num_hidden1_dims = 20,
                 num_hidden2_dims = 10,
                 num_hidden3_dims = 10)
    state_dict = torch.load(path + "model/paper/model_state_%s_10T.pth" % QoIs)
    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model = model.cuda()

May10_storm = [datetime.datetime(2024, 5, 10, 18, 0)]
upper = 48
lower = 6
station_path = path + '../Input/Gannon_storm/AllStations_Gannon/'
station_file = os.listdir(station_path)
Geospace_path = path + '../Input/Gannon_storm/Geospace_v2/'
Geospace_stations= sorted([x.split('.txt')[0] for x in os.listdir(Geospace_path)])
My_stations = sorted([x.split('.pkl')[0] for x in station_file])
station_to_evaluate = sorted(list(set(Geospace_stations) & set(My_stations)))
station_to_evaluate_file = sorted([x for x in station_file if (x.split('.pkl')[0] in station_to_evaluate)])

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

                            pred_Y_DGP_lr = pred_Y_DGP - qnorm_975*sigma.reshape(-1,1)
                            pred_Y_DGP_up = pred_Y_DGP + qnorm_975*sigma.reshape(-1,1)
                            pred_Y_OG_lr = scalerY.inverse_transform(pred_Y_DGP_lr).flatten()
                            pred_Y_OG_up = scalerY.inverse_transform(pred_Y_DGP_up).flatten()
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
                            plt.savefig(root_path+"/figure/TestSet3/TestSet3_stations_%s.png" % QoIs,
                                        bbox_inches='tight')
                            plt.close()


"""
Preparing QoIs for FFT with uncertainty quantification. We do random sampling to propagate uncertainty in QoIs to
the downstream FFT calculation and geoelectric field integration.
"""
if make_FFTfile:
    os.makedirs(root_path+"/data/test_station/FFT_withUQ/QoIs/", exist_ok=True)
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
                    samples.to_csv(root_path+"/data/test_station/FFT_withUQ/QoIs/" + 
                                        station_to_evaluate_FFTfile[i][0:3]+'_%s.csv' % QoIs, index=False)

if make_FFT_mergedfile:
    os.makedirs(root_path+"/data/test_station/FFT_withUQ/merged/", exist_ok=True)
    for i in range(len(station_to_evaluate_FFT)):
        station_code = station_to_evaluate_FFT[i]
        ## merge files for FFT:
        dBE = pd.read_csv(root_path+"/data/test_station/FFT_withUQ/QoIs/%s_dBE.csv" % station_code)
        dBH = pd.read_csv(root_path+"/data/test_station/FFT_withUQ/QoIs/%s_dBH.csv" % station_code)
        dBN = pd.read_csv(root_path+"/data/test_station/FFT_withUQ/QoIs/%s_dBN.csv" % station_code)
        station_HNE = pd.merge(pd.merge(dBH, dBN), dBE)
        station_HNE = station_HNE[['Year', 'Month', 'Day', 'Hour', 'Min', 'Sec',
                                   'dBH', 'B_EastGeomag', 'B_NorthGeomag', 'realization']]
        station_HNE.to_csv(root_path+'/data/test_station/FFT_withUQ/merged/%s.csv' % station_code)