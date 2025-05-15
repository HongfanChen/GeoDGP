import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import scipy
from rich.progress import track
import datetime
import pickle
import re
import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from src.clean import *
# ---------------------------------------------------------------------------------------------------------------
# --------------------------------------------- Load OMNI ------ ------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
path = root_path+"/data/train/"
OMNI = pd.read_pickle(path + "../Input/OMNI_1995_2022_5m_feature_Storms_50nT.pkl").dropna()
## 5 minutes window median.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
QoIs = "dBH"
station_path = root_path+"/data/AllStations_AllYear_1min_raw/"
window_max = 10
hour = 2
window_nominal = '%sT' % window_max
res_folder = path + 'Maximum_storms/%sHAhead/%sT' % (hour, window_max)
os.makedirs(res_folder, exist_ok=True)
os.makedirs(res_folder + '/%s' % QoIs, exist_ok=True)

# ---------------------------------------------------------------------------------------------------------------
# --------------------------------------------- 1 hour ahead ----------------------------------------------------
OMNI.index = OMNI.index +datetime.timedelta(hours=hour)
validation_date = [datetime.datetime(2002, 10, 7, 7, 30),
                   datetime.datetime(2012, 11, 14, 7, 30),
                   datetime.datetime(1997, 11, 7, 4, 30),
                   datetime.datetime(2005, 1, 18, 8, 30),
                   datetime.datetime(2016, 1, 20, 16, 30),
                   datetime.datetime(2002, 4, 18, 7, 30),
                   datetime.datetime(2011, 10, 25, 1, 30),
                   datetime.datetime(2005, 8, 31, 19, 30),
                   datetime.datetime(2004, 8, 30, 22, 30),
                   datetime.datetime(2006, 12, 15, 7, 30)]
val_storm_date_list = [(storm_date - datetime.timedelta(hours=24),
                        storm_date + datetime.timedelta(hours=36)) for storm_date in validation_date]
### the first one for Dagger comparison, and the remaining for 2015.
test_storm_onset_date = [datetime.datetime(2011, 8, 5, 18, 2),
                         datetime.datetime(2015, 2, 16, 19, 24),
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
test_storm_date_list = [(storm_date - datetime.timedelta(hours=6),
                         storm_date + datetime.timedelta(hours=48)) for storm_date in test_storm_onset_date]
storm_to_remove = val_storm_date_list +test_storm_date_list

# # ## let's calcualte the scaler first.
if not os.path.exists(path + "scaler/scalerY_%s_%s.pkl" % (QoIs, window_max)):
    if not os.path.exists(path + "scaler/scalerX_%s.pkl" % window_max):
        year = 2005
        station_TargetYear_file = sorted([x for x in os.listdir(station_path) if bool(re.match(r'.*_%s\.pkl$' % year, x))])
        station_filepath_List = [station_path + x for x in station_TargetYear_file]
        train_Y = []
        val_Y = []
        train_X = []
        val_X = []
        for i in track(range(len(station_filepath_List))):
            file_path = station_filepath_List[i]
            train_X_tmp, val_X_tmp, train_Y_tmp, val_Y_tmp = train_val_split_windowMax(file_path, OMNI, 
                                                                                       val_storm_date_list,
                                                                                       storm_to_remove,
                                                                                       window_nominal,
                                                                                       QoIs)
            train_X.append(train_X_tmp)
            val_X.append(val_X_tmp)
            train_Y.append(train_Y_tmp)
            val_Y.append(val_Y_tmp)

        train_X = pd.concat(train_X, axis=0)
        train_Y = pd.concat(train_Y, axis=0)
        train_Y = train_Y.to_numpy().reshape(-1,1)

        scalerX = StandardScaler()
        scalerY = StandardScaler()
        ## fit scaler
        scalerX.fit(train_X.to_numpy())
        with open(path + "scaler/scalerX_%s.pkl" % window_max, 'wb') as f:
            pickle.dump(scalerX, f)
        scalerY.fit(train_Y)
        with open(path + "scaler/scalerY_%s_%s.pkl" % (QoIs, window_max), 'wb') as f:
            pickle.dump(scalerY, f)
    else:
        year = 2005
        station_TargetYear_file = sorted([x for x in os.listdir(station_path) if bool(re.match(r'.*_%s\.pkl$' % year, x))])
        station_filepath_List = [station_path + x for x in station_TargetYear_file]
        train_Y = []
        val_Y = []
#         train_X = []
#         val_X = []
        for i in track(range(len(station_filepath_List))):
            file_path = station_filepath_List[i]
            train_X_tmp, val_X_tmp, train_Y_tmp, val_Y_tmp = train_val_split_windowMax(file_path, OMNI, 
                                                                                       val_storm_date_list,
                                                                                       storm_to_remove,
                                                                                       window_nominal,
                                                                                       QoIs)
#             train_X.append(train_X_tmp)
#             val_X.append(val_X_tmp)
            train_Y.append(train_Y_tmp)
            val_Y.append(val_Y_tmp)

#         train_X = pd.concat(train_X, axis=0)
        train_Y = pd.concat(train_Y, axis=0)
        train_Y = train_Y.to_numpy().reshape(-1,1)

#         scalerX = StandardScaler()
        scalerY = StandardScaler()
        ## fit scaler
#         scalerX.fit(train_X.to_numpy())
#         with open(path + "scalerX_%s.pkl" % window_max, 'wb') as f:
#             pickle.dump(scalerX, f)
        scalerY.fit(train_Y)
        with open(path + "scaler/scalerY_%s_%s.pkl" % (QoIs, window_max), 'wb') as f:
            pickle.dump(scalerY, f)

AvailableYear = [date.year for date in OMNI.index]
AvailableYear = list(set(AvailableYear))
AvailableYear.remove(2015)
## load scaler from dBH 20min maximum and OMNI 5min project
with open(path + "scaler/scalerX_%s.pkl" % window_max,'rb') as f:
    scalerX = pickle.load(f)
with open(path + "scaler/scalerY_%s_%s.pkl" % (QoIs, window_max),'rb') as f:
    scalerY = pickle.load(f)
memory_usage = 0
num = 0
start_year = AvailableYear[0]
end_year = AvailableYear[-1]
val_X = []
val_Y = []
train_X = []
train_Y = []
for year in track(AvailableYear):
    station_TargetYear_file = sorted([x for x in os.listdir(station_path) if bool(re.match(r'.*_%s\.pkl$' % year, x))])
    station_filepath_List = [station_path + x for x in station_TargetYear_file]
    for i in range(len(station_filepath_List)):
        file_path = station_filepath_List[i]
        train_X_tmp, val_X_tmp, train_Y_tmp, val_Y_tmp = train_val_split_windowMax(file_path, OMNI,
                                                                                   val_storm_date_list,
                                                                                   storm_to_remove,
                                                                                   window_nominal,
                                                                                   QoIs)
        train_X.append(train_X_tmp)
        val_X.append(val_X_tmp)
        train_Y.append(train_Y_tmp)
        val_Y.append(val_Y_tmp)
        memory_usage += train_X_tmp.memory_usage(index=True, deep=False).sum() /1024/1024
        print("year: %s, %03d/%03d, Memory Used: %.1f MB" % (year, i+1, len(station_filepath_List), memory_usage))
    if year == end_year:
        val_X = standardizeX_and_to_torch32_fromList(val_X, scalerX)
        val_Y = standardizeY_and_to_torch32_fromList(val_Y, scalerY)
        torch.save(val_X, res_folder + '/val_X_%s.pt' % QoIs)
        torch.save(val_Y, res_folder + '/val_Y_%s.pt' % QoIs)
    if (memory_usage < 25*1024) & (year<end_year):
        continue
    else:
        num = num + 1
        memory_usage = 0
        train_X = standardizeX_and_to_torch32_fromList(train_X, scalerX)
        train_Y = standardizeY_and_to_torch32_fromList(train_Y, scalerY)
        torch.save(train_X, res_folder + '/train_X_%s_%03d.pt'% (QoIs, num))
        torch.save(train_Y, res_folder + '/%s/train_Y_%03d.pt' % (QoIs, num))
        print("Training file %03d saved" % num)
        train_X = []
        train_Y = []
