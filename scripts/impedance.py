
import bezpy
import numpy as np
import matplotlib.pyplot as plt
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
def load_site(path, file_name):
    site = bezpy.mt.read_xml(path+file_name)
    site_pd = site.data
#     periods = site_pd.index
    site_Z_slice = site_pd[['z_zxx', 'z_zxy', 'z_zyx', 'z_zyy']]
    site_Z = site_Z_slice.copy()
    # Calculate phase angles and create new columns
    for col in site_Z.columns:
        site_Z[f'{col}_phase'] = np.angle(site_Z[col], deg=True)
    return site_Z

path = root_path + "/data/MT_sites/"
station = 'OTT'
filenames = ['USArray.NYE56.2016.r1.xml', 'USArray.NYE57.2016.r1.xml']
site_IDs = ['NYE56', 'NYE57']
site_data = [load_site(path, file_name) for file_name in filenames]
periods = site_data[0].index.values
omega = 2*np.pi / periods

line_width = 3
line_width_reference = 4
font_legend = 16
font_ticks = 16
plt.rcParams["figure.figsize"] = [10, 13]
plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({'font.size': 20})
plt.subplot(2,1,1)
periods = site_data[0].index.values
omega = 2*np.pi / periods
plt.plot(omega, site_data[0]['z_zxx'].apply(abs), label='xx', linewidth=line_width)
plt.plot(omega, site_data[0]['z_zxy'].apply(abs), label='xy', linewidth=line_width)
plt.plot(omega, site_data[0]['z_zyx'].apply(abs), label='yx', linewidth=line_width)
plt.plot(omega, site_data[0]['z_zyy'].apply(abs), label='yy', linewidth=line_width)
plt.plot(omega, omega**(0.5), label='$\omega^{0.5}$',
         color='black', linestyle='dashed', linewidth=line_width_reference)
plt.legend(fontsize = font_legend)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('$\omega$ [Hz]')
plt.ylabel('Z [(mV/km)/(nT)]')
plt.title("MT Site ID: " + site_IDs[0])
plt.xticks(fontsize = font_ticks)
plt.yticks(fontsize = font_ticks)
plt.subplot(2,1,2)
periods = site_data[1].index.values
omega = 2*np.pi / periods
plt.plot(omega, site_data[1]['z_zxx'].apply(abs), label='xx', linewidth=line_width)
plt.plot(omega, site_data[1]['z_zxy'].apply(abs), label='xy', linewidth=line_width)
plt.plot(omega, site_data[1]['z_zyx'].apply(abs), label='yx', linewidth=line_width)
plt.plot(omega, site_data[1]['z_zyy'].apply(abs), label='yy', linewidth=line_width)
plt.plot(omega, omega**(0.5), label='$\omega^{0.5}$',
         color='black', linestyle='dashed', linewidth=line_width_reference)
plt.legend(fontsize = font_legend)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('$\omega$ [Hz]')
plt.ylabel('Z [(mV/km)/(nT)]')
plt.title("MT Site ID: " + site_IDs[1])
plt.xticks(fontsize = font_ticks)
plt.yticks(fontsize = font_ticks)
plt.savefig(root_path+'/figure/TestSet3/MT_SITE.png', facecolor = 'white',bbox_inches='tight')
# %%
