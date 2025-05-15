import os
import load_stations
import multiprocessing as mp
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

# data can be downloaded from https://supermag.jhuapl.edu/info/, go to download button,
## select a year and download One-Year-All-Stations. We used data from 1995-2022: all_stations_allYYYY.netcdf
file_dir = root_path + "/data/AllStations_AllYear/"
file_name_list = os.listdir(file_dir)
file_path_list = [file_dir + YearData for YearData in file_name_list]
output_dir = root_path + "/data/AllStations_AllYear_1min_raw/"
if __name__ =='__main__':
    pool = mp.Pool(2)
    pool.starmap(load_stations.load_AllStations_YearbyYear,
                 zip(
                     (file_path_list,)*len(file_path_list),
                     (file_name_list,)*len(file_path_list),
                     list(range(len(file_path_list))),
                     (output_dir,)*len(file_path_list)
                    )
                )
    pool.close()


