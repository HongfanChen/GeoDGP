import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import scipy
import datetime
from sklearn.metrics import mean_absolute_error
import pickle
from torch.utils.data import DataLoader
import re
import matplotlib.dates as mdates
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from src.utils import *
from src.model import *
    
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

## 5 minutes window median.
path = root_path+ "/data/train/"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
yeojohnson_lmbda = 1
window_max = 10
make_TestSet2_figure = True
make_TestSet2 = False
hour = 1
station_to_visualize = sorted(['ABK', 'NEW', 'OTT', 'MEA', 'WNG', 'YKC'])
torch.manual_seed(20241029)



if make_TestSet2:
    for year in [2011, 2015]:
        for QoIs in ['dBN', 'dBE', 'dBH']:
            with open(path + "scaler/scalerX_%s.pkl" % window_max,'rb') as f:
                scalerX = pickle.load(f)
            with open(path + "scaler/scalerY_%s_%s.pkl" % (QoIs, window_max),'rb') as f:
                scalerY = pickle.load(f)
            X_shape = 96
            ## set results path
            res_path = root_path+"/data/test_station/TestSet2/%sT/%s/" % (window_max, QoIs)
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
            model = DeepGaussianProcess(X_shape,
                                DEVICE,
                 num_hidden1_dims = 20,
                 num_hidden2_dims = 10,
                 num_hidden3_dims = 10)
            state_dict_ahead = torch.load(path + "model/paper/model_state_%s_10T_%sh_ahead.pth" % (QoIs, hour))
            model.load_state_dict(state_dict_ahead)
            ## model persistence
            model_p = DeepGaussianProcess(X_shape,
                                DEVICE,
                 num_hidden1_dims = 20,
                 num_hidden2_dims = 10,
                 num_hidden3_dims = 10)
            state_dict = torch.load(path + "model/paper/model_state_%s_10T.pth" % QoIs)
            model_p.load_state_dict(state_dict)
            if torch.cuda.is_available():
                model = model.cuda()
                model_p = model_p.cuda()
            
            ## load station filenames
            # QoIs_SM_20min_SM_lon_lat_1995_2002
            # station_path = path + '../QoIs_SM_20min_SM_lon_lat_1995_2002/'
            station_path = root_path+"/data/AllStations_AllYear_1min_raw/"
            station_file = sorted([x for x in os.listdir(station_path) if bool(re.match(r'.*_%s\.pkl$' % year, x))])
            dagger_time = np.load(root_path+'/data/DAGGER/%s/dagger_time_%s.npy' % (year,year))
            dagger_stations = np.load(root_path+'/data/DAGGER/%s/dagger_stations_%s.npy' % (year,year))
            dagger_pred = np.load(root_path+'/data/DAGGER/%s/dagger_%s_%s.npy' % (year, QoIs, year))
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
                    # Deep GP model predictions: --------------------------------------------------------------------------
                    file_path = station_path + station_to_evaluate_file[i]
                    station_data = pd.read_pickle(file_path)
                    station_data = station_data.dropna()
                    station_data.index = station_data['Time']
                    station_data = station_data[[QoIs, "SM_lon", "SM_lat"]]
                    storm_X, storm_Y = test_general_matchDAGGER(file_path, OMNI, QoIs,
                                                                              storm_start, storm_end)
                    dagger_station_idx = np.where(dagger_stations == station_to_evaluate_file[i].split('_%s.pkl' % year)[0])[0].item()
                    dagger_pd = pd.DataFrame(dagger_pred[:,dagger_station_idx])
                    dagger_pd.index = dagger_time
                    dagger_pd.columns = ['%sDagger'% QoIs]
                    Joint_index = storm_Y.index.intersection(dagger_pd.index)
                    Joint_index = station_data.index.intersection(Joint_index-deltaT-deltaT_SW)+deltaT+deltaT_SW
                    storm_X = storm_X.loc[Joint_index]
                    storm_Y = storm_Y.loc[Joint_index]
                    if not storm_X.shape[0] == 0:
                        ## 1-h prediction
                        storm_Y_OG, pred_Y_OG, pred_Y_OG_lr, pred_Y_OG_up = DGP_pred(storm_X,storm_Y,model)
                        ## model persistence
                        storm_Y_OG_p, pred_Y_OG_p, pred_Y_OG_lr_p, pred_Y_OG_up_p = DGP_pred(storm_X,storm_Y,model_p)
                        ## the observation persistence
                        observation_persistence = station_data.loc[Joint_index-deltaT-deltaT_SW][QoIs].to_numpy().reshape(-1,1)
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

if make_TestSet2_figure:
    for year in [2011]:
        for QoIs in ['dBH']:
            with open(path + "scaler/scalerX_%s.pkl" % window_max,'rb') as f:
                scalerX = pickle.load(f)
            with open(path + "scaler/scalerY_%s_%s.pkl" % (QoIs, window_max),'rb') as f:
                scalerY = pickle.load(f)
            X_shape = 96
            ## set results path
            res_path = root_path+"/data/test_station/TestSet2/%sT/%s/" % (window_max, QoIs)
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
            station_path = root_path+"/data/AllStations_AllYear_1min_raw/"
            station_file = sorted([x for x in os.listdir(station_path) if bool(re.match(r'.*_%s\.pkl$' % year, x))])
            dagger_time = np.load(root_path+'/data/DAGGER/%s/dagger_time_%s.npy' % (year,year))
            dagger_stations = np.load(root_path+'/data/DAGGER/%s/dagger_stations_%s.npy' % (year,year))
            dagger_pred = np.load(root_path+'/data/DAGGER/%s/dagger_%s_%s.npy' % (year, QoIs, year))
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

                        # Deep GP model predictions: --------------------------------------------------------------------------
                        file_path = station_path + station_to_evaluate_file[i]
                        storm_X, storm_Y = test_general_matchDAGGER(file_path, OMNI, QoIs,
                                                                                  storm_start, storm_end)
                        dagger_station_idx = np.where(dagger_stations == 
                                                      station_to_evaluate_file[i].split('_%s.pkl' % year)[0])[0].item()
                        dagger_pd = pd.DataFrame(dagger_pred[:,dagger_station_idx])
                        dagger_pd.index = dagger_time
                        dagger_pd.columns = ['%sDagger'% QoIs]
                        Joint_index = storm_Y.index.intersection(dagger_pd.index)
                        storm_X = storm_X.loc[Joint_index]
                        storm_Y = storm_Y.loc[Joint_index]
                        if not storm_X.shape[0] == 0:
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
                                plt.savefig(root_path+"/figure/TestSet2/TestSet2_stations_%s_%s.png" % (QoIs, year),
                                            bbox_inches='tight', dpi=300)
                                plt.close()