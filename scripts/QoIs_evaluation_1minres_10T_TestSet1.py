import torch
import pandas as pd
import numpy as np
import os
import scipy
from rich.progress import track
import datetime
from sklearn.metrics import mean_absolute_error
import pickle
from torch.utils.data import DataLoader
import re
from scipy.stats import percentileofscore
import astropy
import astropy.units as u
from sunpy.coordinates import frames
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from src.utils import *
from src.model import *
    
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
        pred_Y_DGP = pred_means.mean(0).cpu().numpy().reshape(-1,1)
        storm_Y_OG = scalerY.inverse_transform(storm_loader.dataset.Y.cpu().numpy().reshape(-1,1))
        pred_Y_OG = scalerY.inverse_transform(pred_Y_DGP)
    return storm_Y_OG, pred_Y_OG

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

## 5 minutes window median.
path = root_path + "/data/train/"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
torch.manual_seed(20241023)

if WithDst:
    OMNI = pd.read_pickle(path + "../Input/OMNI_paper_5m_feature.pkl").dropna()
else:
    OMNI = pd.read_pickle(path + "../Input/OMNI_paper_5m_feature.pkl").dropna()
    OMNI.drop('Dst', axis=1, inplace=True)
    OMNI.drop(columns=OMNI.columns[80:93], axis=1, inplace=True)
deltaT_SW = datetime.timedelta(minutes=45)
if multihour_ahead:
    model = DeepGaussianProcess(X_shape,
                                DEVICE,
                 num_hidden1_dims = 20,
                 num_hidden2_dims = 10,
                 num_hidden3_dims = 10)
    model_ahead = DeepGaussianProcess(X_shape,
                                DEVICE,
                 num_hidden1_dims = 20,
                 num_hidden2_dims = 10,
                 num_hidden3_dims = 10)
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
    model = DeepGaussianProcess(X_shape,
                                DEVICE,
                 num_hidden1_dims = 20,
                 num_hidden2_dims = 10,
                 num_hidden3_dims = 10)
    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model = model.cuda()

year = 2015
station_path = root_path+"/data/AllStations_AllYear_1min_raw/"
station_file = sorted([x for x in os.listdir(station_path) if bool(re.match(r'.*_%s\.pkl$' % year, x))])
Geospace_path = root_path+"/data/Geospace_Qusai/"
Geospace_stations= sorted([x.split('.csv')[0] for x in os.listdir(Geospace_path)])
My_stations = sorted([x.split('_2015')[0] for x in station_file])
station_to_evaluate = sorted(list(set(Geospace_stations) & set(My_stations)))
station_to_evaluate_file = sorted([x for x in station_file if (x.split('_2015')[0] in station_to_evaluate)])
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
    station_info = pd.read_csv(root_path + "/data/Input/station_info.csv")
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
        station_data = station_data[[QoIs, "SM_lon", "SM_lat"]]
        station_storm_df = []
        for j in range(len(test_storm_date_list)):
            storm_onset = test_storm_date_list[j]
            storm_start = storm_onset - datetime.timedelta(hours = lower)
            storm_end = storm_onset + datetime.timedelta(hours = upper)
            station_storm_df.append(station_data.loc[storm_start:storm_end].drop_duplicates())
        station_storm_df = pd.concat(station_storm_df, axis=0)
        Joint_index = OMNI.index.intersection(station_storm_df.index)
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
        res_path = root_path+"/data/test_station/TestSet1/%sT_%sh_ahead/%s/" % (window_max, hour, QoIs)
    else:
        res_path = root_path+"/data/test_station/TestSet1/%sT/%s/" % (window_max, QoIs)
    score_df.to_csv(res_path+"interval_score.csv", index=False)
    coverage_df.to_csv(res_path+"coverage.csv", index=False)
    width_df.to_csv(res_path+"interval_width.csv", index=False)

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
        res_path = root_path+"/data/test_station/TestSet1/%sT_%sh_ahead/%s/" % (window_max, hour, QoIs)
    else:
        if WithDst:
            res_path = root_path+"/data/test_station/TestSet1/%sT/%s/" % (window_max, QoIs)
        else:
            res_path = root_path+"/data/test_station/TestSet1/NoDst/%sT/%s/" % (window_max, QoIs)
    os.makedirs(res_path, exist_ok = True)
    HSS_df.to_csv(res_path+"HSS.csv", index=False)
