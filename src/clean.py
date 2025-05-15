import torch
import pandas as pd
import numpy as np
def train_val_split_windowMax(file_path, OMNI_data, Storm_to_validate, Storm_to_remove, window = '20T',
                    QoIs = "dBH_dt"):
    station_data = pd.read_pickle(file_path)
    station_data.index = station_data['Time']
    station_data = station_data.dropna()
#     station_max = station_data[QoIs].resample(window, label='left').max()
    station_max = station_data[QoIs].resample(window, label='left').apply(lambda x : max(x, key = abs, default = np.NaN))
    station_data = pd.DataFrame(station_max).join(station_data[["SM_lon", "SM_lat"]], how='left').dropna()
    val_SuperMAG_raw = []
    for i in range(len(Storm_to_validate)):
        start_date = Storm_to_validate[i][0]
        end_date = Storm_to_validate[i][1]
        storm_data = station_data.loc[start_date:end_date]
        val_SuperMAG_raw.append(storm_data)
    val_SuperMAG_raw = pd.concat(val_SuperMAG_raw, axis=0)

    for i in range(len(Storm_to_remove)):
        start_date = Storm_to_remove[i][0]
        end_date = Storm_to_remove[i][1]
        if i == 0:
            non_storm_idx = (station_data.index < start_date) | (station_data.index > end_date)
            train_SuperMAG_raw = station_data[non_storm_idx]
        else:
            non_storm_idx = (train_SuperMAG_raw.index < start_date) | (train_SuperMAG_raw.index > end_date)
            train_SuperMAG_raw = train_SuperMAG_raw[non_storm_idx]

    ## remove missing OMNI data
    train_existing_OMNI = OMNI_data.index.intersection(train_SuperMAG_raw.index)
    val_existing_OMNI = OMNI_data.index.intersection(val_SuperMAG_raw.index)
    ## now find the available OMNI data
    train_OMNI_X = OMNI_data.loc[train_existing_OMNI]
    val_OMNI_X = OMNI_data.loc[val_existing_OMNI]
    ## find superMAG data that match OMNI data
    train_SuperMAG = train_SuperMAG_raw.loc[train_existing_OMNI]
    val_SuperMAG = val_SuperMAG_raw.loc[val_existing_OMNI]
    train_Y_raw = train_SuperMAG[QoIs].reset_index(drop=True, inplace=False).astype(np.float32)
    val_Y_raw = val_SuperMAG[QoIs].reset_index(drop=True, inplace=False).astype(np.float32)
    train_X_raw = pd.concat([train_OMNI_X,
               train_SuperMAG[["SM_lat"]],
              np.cos(train_SuperMAG["SM_lon"].astype(np.float32) / 180 * np.pi),
              np.sin(train_SuperMAG["SM_lon"].astype(np.float32) / 180 * np.pi)],
              axis = 1).reset_index(drop=True, inplace=False).astype(np.float32)
    val_X_raw = pd.concat([val_OMNI_X,
                           val_SuperMAG[["SM_lat"]],
                           np.cos(val_SuperMAG["SM_lon"].astype(np.float32) / 180 * np.pi),
                           np.sin(val_SuperMAG["SM_lon"].astype(np.float32) / 180 * np.pi)],
                           axis = 1).reset_index(drop=True, inplace=False).astype(np.float32)
    train_X_raw.columns = train_X_raw.columns[:-2].to_list() + ["CosLon", "SinLon"]
    val_X_raw.columns = train_X_raw.columns
    return train_X_raw, val_X_raw, train_Y_raw, val_Y_raw

def standardizeX_and_to_torch32_fromList(train_X, scalerX):
    train_X = pd.concat(train_X, axis=0)
    train_X = scalerX.transform(train_X.to_numpy())
    train_X = torch.from_numpy(train_X).type(torch.float32)
    return train_X
def standardizeY_and_to_torch32_fromList(train_Y, scalerY):
    train_Y = pd.concat(train_Y, axis=0)
    train_Y = train_Y.to_numpy().reshape(-1,1)
    train_Y = scalerY.transform(train_Y)
    train_Y = torch.from_numpy(train_Y).type(torch.float32)
    train_Y = train_Y.reshape(train_Y.shape[0])
    return train_Y