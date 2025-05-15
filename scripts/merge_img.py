# %%
import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from src.utils import *
## Test Set 1
path = root_path+"/figure/TestSet1/"
os.makedirs(path, exist_ok=True)
file_HSS_test1 = [path + x for x in ["TestSet1_HSS_200_NorthPole_dBH.png",
                                     "TestSet1_HSS_200_SouthPole_dBH.png"]]
combine_images(columns=2, space=0, images=file_HSS_test1, save_path=path+'TestSet1_HSS_200_dBH.png')
file_HSS_test1 = [path + x for x in ["TestSet1_HSS_50_dBH.png",
                                     "TestSet1_HSS_200_dBH.png"]]
combine_images(columns=1, space=0, images=file_HSS_test1, save_path=path+'TestSet1_HSS_globalMap.png')
## Test Set 3
path = root_path+'/figure/TestSet3/'
os.makedirs(path, exist_ok=True)
file_globalmap = []
file_polarmap = []
file_globalmap.append(path + 'GeoDGP_global.png')
file_globalmap.append(path + 'Geospace_global.png')
file_polarmap.append(path + 'AMPERE_FACs.png')
file_polarmap.append(path + 'GeoDGP_Polar.png')
file_polarmap.append(path + 'Geospace_Polar.png')
combine_images(columns=1, space=25, images=file_globalmap, save_path=path+'globalMap.png')
combine_images(columns=1, space=0, images=file_polarmap, save_path=path+'polarMap.png')