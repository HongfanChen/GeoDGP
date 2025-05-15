## This script is used to reduce the size of the data by removing non-storm data.

import pandas as pd
import os
import re
import datetime
from rich.progress import track
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

year = 2015
station_path = root_path+"/data/AllStations_AllYear_1min_raw/"
station_file = sorted([x for x in os.listdir(station_path) if bool(re.match(r'.*_%s\.pkl$' % year, x))])
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


for i in track(range(len(station_file))):
    file_path = station_path + station_file[i]
    station_data = pd.read_pickle(file_path)
    station_data = station_data.dropna()
    station_data.index = station_data['Time']
    station_storm = []
    for j in range(len(test_storm_date_list)):
        storm_onset = test_storm_date_list[j]
        storm_start = storm_onset - datetime.timedelta(hours = lower)
        storm_end = storm_onset + datetime.timedelta(hours = upper)
        station_storm.append(station_data[storm_start:storm_end])
    station_storm = pd.concat(station_storm, axis=0).drop_duplicates()
    station_storm.to_pickle(root_path+"/data/AllStations_AllYear_1min_raw/"+station_file[i])

year=2011
station_path = root_path+"/data/AllStations_AllYear_1min_raw/"
station_file = sorted([x for x in os.listdir(station_path) if bool(re.match(r'.*_%s\.pkl$' % year, x))])
test_storm_date_list = [datetime.datetime(2011, 8, 5, 18, 2)]
upper = 48
lower = 12

for i in track(range(len(station_file))):
    file_path = station_path + station_file[i]
    station_data = pd.read_pickle(file_path)
    station_data = station_data.dropna()
    station_data.index = station_data['Time']
    station_storm = []
    for j in range(len(test_storm_date_list)):
        storm_onset = test_storm_date_list[j]
        storm_start = storm_onset - datetime.timedelta(hours = lower)
        storm_end = storm_onset + datetime.timedelta(hours = upper)
        station_storm.append(station_data[storm_start:storm_end])
    station_storm = pd.concat(station_storm, axis=0).drop_duplicates()
    station_storm.to_pickle(root_path+"/data/AllStations_used_paper/"+station_file[i])
    
test_storm_date_list = [datetime.datetime(2011, 8, 5, 18, 2),
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
upper = 48
lower = 24

OMNI = pd.read_pickle(root_path+"/data/Input/OMNI_1995_2022_5m_feature.pkl")
OMNI_storm = []
for j in range(len(test_storm_date_list)):
    storm_onset = test_storm_date_list[j]
    storm_start = storm_onset - datetime.timedelta(hours = lower)
    storm_end = storm_onset + datetime.timedelta(hours = upper)
    OMNI_storm.append(OMNI[storm_start:storm_end])
OMNI_storm = pd.concat(OMNI_storm, axis=0).drop_duplicates()
OMNI_storm.to_pickle(root_path+"/data/Input/OMNI_paper_5m_feature.pkl")




