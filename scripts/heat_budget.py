# Python script to quantify the effect of sensor shielding on the heat budget of a small pond (Eawag pond)

import pandas as pd
import numpy as np
import os

""" Matplotlib configuration """
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
from datetime import datetime, timedelta

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Choose "all" to plot all data (used for scatter plot in manuscript) and "june" to plot only june data (used for time series plot in manuscript)
period = 'all'

# Choose True if a pond with typical mid-latitude temperatures should be used; choose False if the original data of the pond should be used
add_10_degrees = True

# Read in solar radiation data from MeteoSwiss station in Lucerne
meteo = pd.read_csv('./data/meteo/MeteoSwiss_LUZ_data.txt', header=0, skiprows=2, delimiter=';')
meteo['Time'] = pd.to_datetime(meteo['time'], format='%Y%m%d%H%M')
meteo['Timestamp'] = meteo['Time'].apply(lambda x: datetime.timestamp(x))

data_folder = "./data/pond/"
files = os.listdir(data_folder)

if period == 'all':
    result_folder = "./figures/heat_all/"
else:
    result_folder = "./figures/heat_june/"

##### Read sensor data #####

# Hobo sensor with shield
hobo_shield_file = [x for x in files if "hobo_pond_shield" in x]
hobo_shield = pd.read_csv(data_folder + hobo_shield_file[0], header=0, skiprows=0, delimiter=',')
hobo_shield['Time'] = pd.to_datetime(hobo_shield['Date-Time (UTC Standard Time)'], format='%m/%d/%Y %H:%M:%S')
hobo_shield['Timestamp'] = hobo_shield['Time'].apply(lambda x: datetime.timestamp(x))
hobo_shield['Ch: 1 - Temperature Avg : Avg (°C )'] = hobo_shield['Ch: 1 - Temperature Avg : Avg (°C )'] + 0.005 # Calibration of Fall 2022

# Hobo sensor without shield
hobo_noshield_file = [x for x in files if "hobo_pond_noshield" in x]
hobo_noshield = pd.read_csv(data_folder + hobo_noshield_file[0], header=0, skiprows=0, delimiter=',')
hobo_noshield['Time'] = pd.to_datetime(hobo_noshield['Date-Time (UTC Standard Time)'], format='%m/%d/%Y %H:%M:%S')
hobo_noshield['Timestamp'] = hobo_noshield['Time'].apply(lambda x: datetime.timestamp(x))
hobo_noshield['Ch: 1 - Temperature Avg : Avg (°C )'] = hobo_noshield['Ch: 1 - Temperature Avg : Avg (°C )'] + 0.035 # Calibration of Fall 2022

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

# Take rolling average of meteorological variables to reduce scatter
window_size = 6

timestamp = np.arange(start_timestamp,end_timestamp,600)
time = [datetime.fromtimestamp(t) for t in timestamp]
tair = np.interp(timestamp,meteo['Timestamp'],meteo['tre200s0'].rolling(window=window_size,center=True).mean())
swr = np.interp(timestamp,meteo['Timestamp'],meteo['gre000z0'].rolling(window=window_size,center=True).mean())
wind = np.interp(timestamp,meteo['Timestamp'],meteo['fkl010z0'].rolling(window=window_size,center=True).mean())
vap_atm = np.interp(timestamp,meteo['Timestamp'],meteo['pva200s0'].rolling(window=window_size,center=True).mean())

temp_hs = np.interp(timestamp,hobo_shield['Timestamp'],hobo_shield['Ch: 1 - Temperature Avg : Avg (°C )'])
temp_hn = np.interp(timestamp,hobo_noshield['Timestamp'],hobo_noshield['Ch: 1 - Temperature Avg : Avg (°C )'])

if add_10_degrees:
    temp_hs = temp_hs + 10
    temp_hn = temp_hn + 10

label_font = 15

#########################################
############### Plots ###################
#########################################

# Meteo variables
fig0 = plt.figure(figsize=(12,8))
ax1 = fig0.add_subplot(4,1,1)
ax1.plot(time,tair)
ax1.set_ylabel('Air temperature [°C]', fontsize=label_font)

ax2 = fig0.add_subplot(4,1,2)
ax2.plot(time,swr)
ax2.set_ylabel('Solar radiation [W m$^{-2}$]', fontsize=label_font)

ax3 = fig0.add_subplot(4,1,3)
ax3.plot(time,wind)
ax3.set_ylabel('Wind velocity [m s$^{-1}$]', fontsize=label_font)

ax4 = fig0.add_subplot(4,1,4)
ax4.plot(time,vap_atm)
ax4.set_ylabel('Vapor pressure [hPa]', fontsize=label_font)


# Outgoing LWR
fig1 = plt.figure(figsize=(12,8))
ax1 = fig1.add_subplot(4, 1, 1)

emiss_water = 0.97
K_ST = 5.67e-8 # Stefan Boltzmann constant

lwr_hs = -emiss_water*K_ST*((273.15 + temp_hs)**4)
lwr_hn = -emiss_water*K_ST*((273.15 + temp_hn)**4)

ax1.plot(time, lwr_hs, label='Shielded')
ax1.plot(time, lwr_hn, label='Non-shielded')
#ax1.set_xlabel('Date', fontsize=label_font)
ax1.set_ylabel('OLR [W m$^{-2}$]', fontsize=label_font)
ax1.legend()
plt.grid()

# Sensible heat flux
ax2 = fig1.add_subplot(4, 1, 2)

p_air = 960 # Air pressure in mbar (assumed constant)
B0 = 0.665*0.001*p_air # Bowen constant

exponent = 0.1

fu_term = (temp_hs - tair)/(1 - 0.378*vap_atm/p_air)
fu_hs = np.sqrt((2.7*fu_term.clip(min=0)**exponent)**2 + (0.6072*3.1*wind)**2)

fu_term = (temp_hn - tair)/(1 - 0.378*vap_atm/p_air)
fu_hn = np.sqrt((2.7*fu_term.clip(min=0)**exponent)**2 + (0.6072*3.1*wind)**2)

sens_hs = -B0*fu_hs*(temp_hs - tair)
sens_hn = -B0*fu_hn*(temp_hn - tair)

ax2.plot(time, sens_hs, label='Shielded')
ax2.plot(time, sens_hn, label='Non-shielded')
#ax2.set_xlabel('Date', fontsize=label_font)
ax2.set_ylabel('Sensible [W m$^{-2}$]', fontsize=label_font)
ax2.legend()
plt.grid()

# Latent heat flux
ax3 = fig1.add_subplot(4, 1, 3)

# Water vapor saturation pressure in air at water temperature (according to Gill, 1992)
vap_wat_hs = 10**((0.7859 + 0.03477*temp_hs)/(1 + 0.00412*temp_hs))
vap_wat_hs = vap_wat_hs*(1 + 1e-6*p_air*(4.5 + 0.00006*temp_hs**2))

vap_wat_hn = 10**((0.7859 + 0.03477*temp_hn)/(1 + 0.00412*temp_hn))
vap_wat_hn = vap_wat_hn*(1 + 1e-6*p_air*(4.5 + 0.00006*temp_hn**2))

# Calculation of Latent heat flux
latent_hs = -fu_hs*(vap_wat_hs - vap_atm)
latent_hn = -fu_hn*(vap_wat_hn - vap_atm)

ax3.plot(time, latent_hs, label='Shielded')
ax3.plot(time, latent_hn, label='Non-shielded')
#ax3.set_xlabel('Date', fontsize=label_font)
ax3.set_ylabel('Latent [W m$^{-2}$]', fontsize=label_font)
ax3.legend()
plt.grid()

# heat balance
ax4 = fig1.add_subplot(4, 1, 4)
ax4.plot(time, lwr_hs + sens_hs + latent_hs, label='Shielded')
ax4.plot(time, lwr_hn + sens_hn + latent_hn, label='Non-shielded')
ax4.set_xlabel('Date', fontsize=label_font)
ax4.set_ylabel('Balance [W m$^{-2}$]', fontsize=label_font)
ax4.legend()
plt.grid()

plt.suptitle('A positive heat flux is directed into the water')

plt.tight_layout()
# Save figure
fig1.savefig(result_folder + 'heat_pond.png',format='png',dpi=300)

#########################################################
################ Heat flux #######################
#########################################################

# Outgoing LWR
fig2 = plt.figure(figsize=(14,5))
ax1 = fig2.add_subplot(1, 1, 1)
ax1.plot(time, lwr_hs - lwr_hn, label='Outgoing longwave',linestyle='dashed',color='#1f78b4')
ax1.set_xlabel('Date', fontsize=label_font)
ax1.set_ylabel('Heat flux difference [W m$^{-2}$]', fontsize=label_font)
plt.grid()
ax1.tick_params(axis='both', which='major', labelsize=label_font)

# Sensible heat flux
#ax2 = fig2.add_subplot(4, 1, 2)
ax1.plot(time, sens_hs - sens_hn, label='Sensible heat',linestyle='dotted',color='#ff7f00')
#ax2.set_xlabel('Date', fontsize=label_font)
#ax2.set_ylabel('Sensible heat flux [W m$^{-2}$]', fontsize=label_font)
#plt.grid()

# Latent heat flux
#ax3 = fig2.add_subplot(4, 1, 3)
ax1.plot(time, latent_hs - latent_hn, label='Latent heat',linestyle='dashdot',color= '#33a02c')
#ax3.set_xlabel('Date', fontsize=label_font)
#ax3.set_ylabel('Latent heat flux [W m$^{-2}$]', fontsize=label_font)
#plt.grid()

# heat balance
#ax4 = fig2.add_subplot(4, 1, 4)
ax1.plot(time, lwr_hs + sens_hs + latent_hs - lwr_hn - sens_hn - latent_hn, label='Heat balance',color='#6a3d9a')
#ax4.set_xlabel('Date', fontsize=label_font)
#ax4.set_ylabel('Heat balance [W m$^{-2}$]', fontsize=label_font)
#plt.grid()

#plt.suptitle('Difference in heat flux (shielded - nonshielded) without incoming short- and longwave radiation', size=label_font)
plt.legend(fontsize=15)
plt.tight_layout()
# Save figure
fig2.savefig(result_folder + 'heat_pond_diff.png',format='png',dpi=300)

fig2b = plt.figure(figsize=(12,4))
ax4 = fig2b.add_subplot(1, 1, 1)
ax4.plot(time, lwr_hs + sens_hs + latent_hs - lwr_hn - sens_hn - latent_hn, label='Shielded')
ax4.set_xlabel('Date', fontsize=18)
ax4.set_ylabel('Heat flux [W m$^{-2}$]', fontsize=18)
plt.suptitle("Heat flux difference between shielded and unshielded sensor")
plt.grid()
plt.tight_layout()
fig2b.savefig(result_folder + 'heat_pond_diff_only1.png',format='png',dpi=300)

####################################################
################# Temperature plots ################
####################################################

fig3 = plt.figure(figsize=(12,8))
ax1 = fig3.add_subplot(2, 1, 1)

ax1.plot(time, temp_hs, label='Shielded')
ax1.plot(time, temp_hn, label='Non-shielded')
ax1.plot(time, tair, label='Air temperature')
ax1.set_xlabel('Date', fontsize=label_font)
ax1.set_ylabel('Temperature [°C]', fontsize=label_font)
ax1.legend(fontsize=18)
plt.grid()

ax2 = fig3.add_subplot(2, 1, 2)

ax2.plot(time, temp_hn - temp_hs, label='Diff')
ax1.set_xlabel('Date', fontsize=label_font)
ax1.set_ylabel('Temperature [°C]', fontsize=label_font)
ax1.legend()
plt.grid()

plt.tight_layout()

# Save figure
fig3.savefig(result_folder + 'water vs air temperature.png',format='png',dpi=300)

print('\n\nNightandDay')
print('Mean overestimation of LWR: ', np.round(np.mean(lwr_hs - lwr_hn),3))
print('Mean overestimation of sensible heat: ', np.round(np.mean(sens_hs - sens_hn),3))
print('Mean overestimation of latent heat: ', np.round(np.mean(latent_hs - latent_hn),3))
print('Mean overestimation of total heat: ', np.round(np.mean(lwr_hs + sens_hs + latent_hs - lwr_hn - sens_hn - latent_hn),3))
print('Proportion of lwr: ', np.round(np.mean(lwr_hs - lwr_hn)/np.mean(lwr_hs + sens_hs + latent_hs - lwr_hn - sens_hn - latent_hn),3))
print('Proportion of sens: ', np.round(np.mean(sens_hs - sens_hn)/np.mean(lwr_hs + sens_hs + latent_hs - lwr_hn - sens_hn - latent_hn),3))
print('Proportion of latent: ', np.round(np.mean(latent_hs - latent_hn)/np.mean(lwr_hs + sens_hs + latent_hs - lwr_hn - sens_hn - latent_hn),3))

print('\n\nDaytime')
print('Mean daytime overestimation of LWR: ', np.round(np.mean(lwr_hs[swr>10] - lwr_hn[swr>10]),3))
print('Mean daytime overestimation of sensible heat: ', np.round(np.mean(sens_hs[swr>10] - sens_hn[swr>10]),3))
print('Mean daytime overestimation of latent heat: ', np.round(np.mean(latent_hs[swr>10] - latent_hn[swr>10]),3))
print('Mean daytime overestimation of total heat: ', np.round(np.mean(lwr_hs[swr>10] + sens_hs[swr>10] + latent_hs[swr>10] - lwr_hn[swr>10] - sens_hn[swr>10] - latent_hn[swr>10]),3))
print('Proportion of daytime lwr: ', np.round(np.mean(lwr_hs[swr>10] - lwr_hn[swr>10])/np.mean(lwr_hs[swr>10] + sens_hs[swr>10] + latent_hs[swr>10] - lwr_hn[swr>10] - sens_hn[swr>10] - latent_hn[swr>10]),3))
print('Proportion of daytime sens: ', np.round(np.mean(sens_hs[swr>10] - sens_hn[swr>10])/np.mean(lwr_hs[swr>10] + sens_hs[swr>10] + latent_hs[swr>10] - lwr_hn[swr>10] - sens_hn[swr>10] - latent_hn[swr>10]),3))
print('Proportion of daytime latent: ', np.round(np.mean(latent_hs[swr>10] - latent_hn[swr>10])/np.mean(lwr_hs[swr>10] + sens_hs[swr>10] + latent_hs[swr>10] - lwr_hn[swr>10] - sens_hn[swr>10] - latent_hn[swr>10]),3))

print('\n\nMax')
print('Max overestimation of LWR: ', np.round(np.max(lwr_hs - lwr_hn),3))
print('Max overestimation of sensible heat: ', np.round(np.max(sens_hs - sens_hn),3))
print('Max overestimation of latent heat: ', np.round(np.max(latent_hs - latent_hn),3))
print('Max overestimation of total heat: ', np.round(np.max(lwr_hs + sens_hs + latent_hs - lwr_hn - sens_hn - latent_hn),3))

#plt.show()