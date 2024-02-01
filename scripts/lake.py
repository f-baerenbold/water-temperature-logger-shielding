# Python script to plot results from Kivu expedition in March 2018

import pandas as pd
import numpy as np
import numpy.ma as ma
import os

""" Matplotlib configuration """
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
from datetime import datetime, timedelta

from sklearn.linear_model import LinearRegression

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Choose "all" to plot all data (used for scatter plot in manuscript) and "june" to plot only june data (used for time series plot in manuscript)
period = 'all'

# Read in solar radiation data from MeteoSwiss station in Lucerne
meteo = pd.read_csv('./data/meteo/MeteoSwiss_LUZ_data.txt', header=0, skiprows=2, delimiter=';', na_values='-',usecols = ['time','gre000z0'])
meteo['time'] = pd.to_datetime(meteo['time'], format='%Y%m%d%H%M')
meteo['timestamp'] = meteo['time'].apply(lambda x: datetime.timestamp(x))

data_folder = "./data/lake/"
files = os.listdir(data_folder)

if period == 'all':
    result_folder = "./figures/lake_all/"
else:
    result_folder = "./figures/lake_june/"

##### Read sensor data #####

# Hobo sensor with shield
hobo_shield_file = [x for x in files if "hobo_lake_shield" in x]
hobo_shield = pd.read_csv(data_folder + hobo_shield_file[0], header=0, skiprows=0, delimiter=',',usecols = [0,1,2])
hobo_shield['time'] = pd.to_datetime(hobo_shield['Date-Time (UTC Standard Time)'], format='%m/%d/%Y %H:%M:%S')
hobo_shield['timestamp'] = hobo_shield['time'].apply(lambda x: datetime.timestamp(x))
hobo_shield['Ch: 1 - Temperature Avg : Avg (°C )'] = hobo_shield['Ch: 1 - Temperature Avg : Avg (°C )'] + 0.061 # Calibration of Fall 2022

# Hobo sensor without shield
hobo_noshield_file = [x for x in files if "hobo_lake_noshield" in x]
hobo_noshield = pd.read_csv(data_folder + hobo_noshield_file[0], header=0, skiprows=0, delimiter=',',usecols = [0,1,2])
hobo_noshield['time'] = pd.to_datetime(hobo_noshield['Date-Time (UTC Standard Time)'], format='%m/%d/%Y %H:%M:%S')
hobo_noshield['timestamp'] = hobo_noshield['time'].apply(lambda x: datetime.timestamp(x))
hobo_noshield['Ch: 1 - Temperature Avg : Avg (°C )'] = hobo_noshield['Ch: 1 - Temperature Avg : Avg (°C )'] + 0.028 # Calibration of Fall 2022

# Vemco sensor with shield
vemco_shield_file = [x for x in files if "vemco_lake_shield" in x]
vemco_shield = pd.read_csv(data_folder + vemco_shield_file[0], header=0, skiprows=7, delimiter=',',usecols = [0,1,2],encoding = "ISO-8859-1")
vemco_shield['time'] = pd.to_datetime(vemco_shield['Date(yyyy-mm-dd)'] + vemco_shield['Time(hh:mm:ss)'], format='%Y-%m-%d%H:%M:%S')
vemco_shield['timestamp'] = vemco_shield['time'].apply(lambda x: datetime.timestamp(x))
vemco_shield['Temperature (°C)'] = vemco_shield['Temperature (°C)'] + 0.028 # Calibration of Fall 2022

# Vemco sensor without shield
vemco_noshield_file = [x for x in files if "vemco_lake_noshield" in x]
vemco_noshield = pd.read_csv(data_folder + vemco_noshield_file[0], header=0, skiprows=7, delimiter=',',usecols = [0,1,2],encoding = "ISO-8859-1")
vemco_noshield['time'] = pd.to_datetime(vemco_noshield['Date(yyyy-mm-dd)'] + vemco_noshield['Time(hh:mm:ss)'], format='%Y-%m-%d%H:%M:%S') - timedelta(hours=2)
vemco_noshield['timestamp'] = vemco_noshield['time'].apply(lambda x: datetime.timestamp(x))
vemco_noshield['Temperature (°C)'] = vemco_noshield['Temperature (°C)'] + 0.035 # Calibration of Fall 2022

# Define start date of analysis
if period == 'all':
    start_time_str = '2022-05-20 00:00:00'
else:
    start_time_str = '2022-06-08 00:00:00'

start_time = datetime.strptime(start_time_str,'%Y-%m-%d %H:%M:%S')
start_timestamp = datetime.timestamp(start_time)

# Define end date of analysis
if period == 'all':
    end_time_str = '2022-09-04 00:00:00'
else:
    end_time_str = '2022-06-14 00:00:00'

end_time = datetime.strptime(end_time_str,'%Y-%m-%d %H:%M:%S')
end_timestamp = datetime.timestamp(end_time)

# Define time arrays with 10 min step
timestamp = np.arange(start_timestamp,end_timestamp,600)
time = [datetime.fromtimestamp(t) for t in timestamp]

# Interpolate solar radiation (swr) and sensor data on same grid
swr = np.interp(timestamp,meteo['timestamp'],meteo['gre000z0'])

temp_hobo_shield = np.interp(timestamp,hobo_shield['timestamp'],hobo_shield['Ch: 1 - Temperature Avg : Avg (°C )'])
temp_hobo_noshield = np.interp(timestamp,hobo_noshield['timestamp'],hobo_noshield['Ch: 1 - Temperature Avg : Avg (°C )'])

temp_vemco_shield = np.interp(timestamp,vemco_shield['timestamp'],vemco_shield['Temperature (°C)'])
temp_vemco_noshield = np.interp(timestamp,vemco_noshield['timestamp'],vemco_noshield['Temperature (°C)'])

label_font = 15

##########################################################
######################  Figure  Hobo  ####################
##########################################################

# Set up the plot
fig1 = plt.figure(figsize=(14,7))

ax1 = fig1.add_subplot(2, 1, 1)
ax1.grid()

if period == 'all':
    fmt = mdates.MonthLocator(interval=1)
else:
    fmt = mdates.DayLocator(interval=2)

ax1.plot(time,temp_hobo_noshield,linestyle='dashed')
ax1.plot(time,temp_hobo_shield)

ax1.set_ylabel('Temperatur [°C]', size= label_font)

ax1.xaxis.set_major_locator(fmt)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax1.tick_params(axis='both', which='major', labelsize=label_font)

ax2 = fig1.add_subplot(2, 1, 2)
ax2.grid()

ax2.plot(time,temp_hobo_noshield - temp_hobo_shield, label='Surface temperature difference')


ax2.set_ylim(-0.2,1.2)

ax2b = ax2.twinx()
ax2b.plot(time,swr, color='darkorange')
ax2b.fill_between(time,swr, color='darkorange', alpha=0.5)
ax2.set_zorder(1)
ax2.patch.set_visible(False)

ax1.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

ax2b.xaxis.set_major_locator(fmt)
ax2b.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
ax2b.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax2.tick_params(axis='both', which='major', labelsize=label_font)
ax2b.tick_params(axis='both', which='major', labelsize=label_font)

fig1.subplots_adjust(hspace=0.45)
fig1.subplots_adjust(wspace=0.45)


# Set labels
ax1.legend(['Non-shielded','Shielded'], fontsize=label_font)
ax1.set_ylabel('Temperature [°C]', size= label_font)
ax2.set_xlabel('Date', size = label_font)
ax2.set_ylabel('Temperature difference [°C]', color = 'blue',size= label_font)
ax2b.set_ylabel('Solar radiation [W m$^{-2}$]',color='darkorange', size= label_font)
fig1.savefig(result_folder + 'Hobo lake.png',format='png',dpi=300)


################################################################
#########################  Figure Vemco  #######################
################################################################

# Set up the plot
fig2 = plt.figure(figsize=(14,7))

ax1 = fig2.add_subplot(2, 1, 1)
ax1.grid()

if period == 'all':
    fmt = mdates.MonthLocator(interval=1)
else:
    fmt = mdates.DayLocator(interval=2)

ax1.plot(time,temp_vemco_noshield,linestyle='dashed')
ax1.plot(time,temp_vemco_shield)

ax1.set_ylabel('Temperatur [°C]', size= label_font)

ax1.xaxis.set_major_locator(fmt)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax1.tick_params(axis='both', which='major', labelsize=label_font)

ax2 = fig2.add_subplot(2, 1, 2)
ax2.grid()

ax2.plot(time,temp_vemco_noshield - temp_vemco_shield, label='Surface temperature difference')


ax2.set_ylim(-0.2,1.2)

ax2b = ax2.twinx()
ax2b.plot(time,swr, color='darkorange')
ax2b.fill_between(time,swr, color='darkorange', alpha=0.5)
ax2.set_zorder(1)
ax2.patch.set_visible(False)

ax1.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

ax2b.xaxis.set_major_locator(fmt)
ax2b.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
ax2b.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax2.tick_params(axis='both', which='major', labelsize=label_font)
ax2b.tick_params(axis='both', which='major', labelsize=label_font)

fig2.subplots_adjust(hspace=0.45)
fig2.subplots_adjust(wspace=0.45)


# Set lables
ax1.legend(['Non-shielded','Shielded'], fontsize=label_font)
ax1.set_ylabel('Temperature [°C]', size= label_font)
ax2.set_xlabel('Date', size = label_font)
ax2.set_ylabel('Temperature difference [°C]', color = 'blue',size= label_font)
ax2b.set_ylabel('Solar radiation [W m$^{-2}$]',color='darkorange', size= label_font)
fig2.savefig(result_folder + 'Vemco lake.png',format='png',dpi=300)


#####################################################
################ Scatter plot #######################
#####################################################

# Define array of hours
time_hours = [x.hour for x in time]

# Moving average
window_size_swr = 6

# Initialize an empty list to store moving averages
df = pd.DataFrame()
df['swr'] = swr

swr_mov = df['swr'].rolling(window=window_size_swr,center=True,min_periods=1).mean()

fig4 = plt.figure(figsize=(15,5))
ax1 = fig4.add_subplot(1, 3, 1)
sct1 = ax1.scatter(swr_mov,temp_hobo_noshield - temp_hobo_shield,c=time_hours,s=5,cmap='Greens')
#ax1.set_xlabel('Incoming solar radiation [W m$^{-2}$]', size= label_font)
ax1.set_ylabel('Temperature excess [°C]', size= label_font)
ax1.set_ylim(-0.2,1.0)
ax1.tick_params(axis='both', which='major', labelsize=label_font)
ax1.set_title('Hobo (lake)',fontsize=label_font)

corr_hobo = round(ma.corrcoef(ma.masked_invalid(swr_mov), ma.masked_invalid(temp_hobo_noshield - temp_hobo_shield))[1,0]*1000)/1000
model_hobo = LinearRegression().fit(np.array(swr_mov).reshape((-1, 1)), np.array(temp_hobo_noshield - temp_hobo_shield))

textstr = 'R$^2$=$%.2f$\nSlope=$%.1e$' % (corr_hobo,model_hobo.coef_[0])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=label_font,
            verticalalignment='top', bbox=props)

ax2 = fig4.add_subplot(1, 3, 2)
sct2 = ax2.scatter(swr_mov,temp_vemco_noshield - temp_vemco_shield,c=time_hours,s=5,cmap='Greens')
#ax2.set_xlabel('Incoming solar radiation [W m$^{-2}$]', size= label_font)
#ax2.set_ylabel('Temperature excess [°C]', size= label_font)
ax2.set_ylim(-0.2,1.0)
ax2.tick_params(axis='both', which='major', labelsize=label_font)
ax2.set_title('Vemco (lake)',fontsize=label_font)

corr_vemco = round(ma.corrcoef(ma.masked_invalid(swr_mov), ma.masked_invalid(temp_vemco_noshield - temp_vemco_shield))[1,0]*1000)/1000
model_vemco = LinearRegression().fit(np.array(swr_mov).reshape((-1, 1)), np.array(temp_vemco_noshield - temp_vemco_shield))

textstr = 'R$^2$=$%.2f$\nSlope=$%.1e$' % (corr_vemco,model_vemco.coef_[0])
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=label_font,
            verticalalignment='top', bbox=props)

cb = plt.colorbar(sct2,ax=ax2)
cb.set_label(label='Hour of Day',size=label_font-1)

plt.tight_layout()
fig4.savefig(result_folder + 'solar correlation lake.png',format='png',dpi=300)

print('Mean overestimation of Hobo: ', np.round(np.mean(temp_hobo_noshield - temp_hobo_shield),3))
print('Mean overestimation of Vemco: ', np.round(np.mean(temp_vemco_noshield - temp_vemco_shield),3))
print('\n')
print('Mean overestimation of Hobo (only daytime): ', np.round(np.mean(temp_hobo_noshield[swr>10] - temp_hobo_shield[swr>10]),3))
print('Mean overestimation of Vemco (only daytime): ', np.round(np.mean(temp_vemco_noshield[swr>10] - temp_vemco_shield[swr>10]),3))
print('\n')
print('Max overestimation of Hobo: ', np.round(np.max(temp_hobo_noshield - temp_hobo_shield),3))
print('Max overestimation of Vemco: ', np.round(np.max(temp_vemco_noshield - temp_vemco_shield),3))
print('\n')
print('Min/mean/max water temperature: ', np.round(np.min(temp_hobo_shield),3), np.round(np.mean(temp_hobo_shield),3), np.round(np.max(temp_hobo_shield),3))
print('\n')