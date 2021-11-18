# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 11:51:07 2021

@author: Anders
"""
#%% Fourier plot template

import pypsa
#pandas package is very useful to work with imported data, time series, matrices ...
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import numpy as np
plt.style.use("bmh")

pathfiles = r'C:/Users/ander/OneDrive - Aarhus Universitet/Maskiningenioer/Kandidat/3. semester/PreProject Master/WP4/Network_files_1hr/'
pathplot = r'C:/Users/ander/OneDrive - Aarhus Universitet/Maskiningenioer/Kandidat/3. semester/PreProject Master/WP4/Plots/Fourier_series/'

flex= 'elec_s_37'  
solar = 'solar+p3-'
time = 'H'
sectors = 'T-H'
lv = '1.0' #, '1.2', '1.5', '2.0'
#year = '2020'
#transition = 'slow'
#dist='dist0.1'
dists = ['dist0.1', 'dist0.5', 'dist1', 'dist2',  'dist10']
years = ['2020', '2030', '2040', '2050']
transitions = ['slow','mid','fast']



i = 0       #start of the outer itterator
j = 0       #start of the inner itterator


index1 = ['Slow','Mid','Fast']
index2 = [2020, 2030, 2040, 2050]

datos = pd.DataFrame(index=pd.MultiIndex.from_product([pd.Series(data=dists, name='name1',),
                                                       pd.Series(data=transitions, name='name2',),
                                                       pd.Series(data=years, name='name3',)]), 
                      columns=pd.Series(data=np.arange(0,8760), name='hour',))
idx = pd.IndexSlice

storagetechs = ['Utility Battery','EV Battery','Home battery','H2 Store','PHS']

#dist = 'dist2'
#year = '2030'
#transition = 'mid'
dists = ['dist2']
years = ['2030']
transitions = ['mid']

for storagetech in storagetechs:
    for dist in dists:
        i=0
        j=0
        for transition in transitions:
            for year in years:
                network_name= (pathfiles+flex+ '_' + 'lv'+ lv + '__' +time+ '-' + sectors+ '-' + solar +dist+'_'+year+transition+'.nc')
                n = pypsa.Network(network_name)
                if storagetech == 'Utility Battery':
                        batterystore=n.stores_t.p.filter(like='battery')
                        batterystore = batterystore[batterystore.columns.drop(list(batterystore.filter(regex='home')))]    # Drop labels that are not needed.
                        batterystore = batterystore[batterystore.columns.drop(list(batterystore.filter(regex='storage')))] # Drop labels that are not needed.
                        batterystore = batterystore[batterystore.columns.drop(list(batterystore.filter(regex='discharge')))] # Drop labels that are not needed.
                        datos.loc[idx[co2_limit , transition , year], :]=np.array(batterystore.sum(1))
                if storagetech == 'EV Battery':
                        if year == '2020':
                            continue
                        else:
                            batterystoreEV=n.stores_t.p.filter(like='battery storage')
                            datos.loc[idx[co2_limit , transition , year], :]=np.array(batterystoreEV.sum(1))
                if storagetech == 'Home battery':
                        batteryhome=n.stores_t.p.filter(like='home battery')
                        datos.loc[idx[co2_limit , transition , year], :]=np.array(batteryhome.sum(1))
                if storagetech == 'H2 Store':
                        H2Store=n.stores_t.p.filter(like='H2 Store')
                        datos.loc[idx[co2_limit , transition , year], :]=np.array(H2Store.sum(1))   
                if storagetech == 'PHS':
                        PHSstore=n.storage_units_t.p.filter(like='PHS')
                        datos.loc[idx[co2_limit , transition , year], :]=np.array(PHSstore.sum(1)) 
                        
                j = j+1
            i = i +1        #adds one itteration to outer counter
            j=0
            plt.style.use('seaborn-ticks')
            plt.rcParams['axes.labelsize'] = 20
            plt.rcParams['xtick.labelsize'] = 18
            plt.rcParams['ytick.labelsize'] = 18
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
        
            plt.figure(figsize=(15, 10))
            #gs1 = gridspec.GridSpec(8, 1)
            #gs1.update(wspace=0.05)
            
            co2_limits=['0.5', '0.2', '0']
            storage_names=['ror'] #,'battery','H2']
            dic_color={'ror':'darkgreen'}
            storage_names=['ror'] #,'battery','H2']
            dic_color={'2020':'olive','2030':'darkgreen','2040':'red','2050':'black'}
            dic_label={'0.5':'50%','0.2':'20%','0':'0%'}
            dic_alpha={'0.5':1,'0.2':1,'0':1}
            dic_linewidth={'0.5':2,'0.2':2,'0':2}
            
            for i,year in enumerate(years):
                ax2 = plt.subplot()    #[4+2*i:6+2*i,0] 
                ax2.set_xlim(0.1,14500)
                ax2.set_ylim(0,1)
                plt.axvline(x=24, color='lightgrey', linestyle='--')
                plt.axvline(x=24*7, color='lightgrey', linestyle='--')
                plt.axvline(x=24*30, color='lightgrey', linestyle='--')
                plt.axvline(x=8760, color='lightgrey', linestyle='--')   
                #ax1.plot(np.arange(0,8760), datos.loc[idx[co2_limit , transition , year], :]/np.max(datos.loc[idx[co2_limit , transition , year], :]), 
                #         color=dic_color[year], alpha=dic_alpha[year], linewidth=dic_linewidth[year],
                #         label='CO$_2$='+dic_label[year])
                #ax1.legend(loc=(0.2, 1.05), ncol=3, shadow=True,fancybox=True,prop={'size':18})
                n_years=1
                t_sampling=1 # sampling rate, 1 data per hour
                x = np.arange(1,8760*n_years, t_sampling) 
                y = np.hstack([np.array(datos.loc[idx[co2_limit , transition , year], :])]*n_years)
                n = len(x)
                y_fft=np.fft.fft(y)/n #n for normalization    
                frq=np.arange(0,1/t_sampling,1/(t_sampling*n))        
                period=np.array([1/f for f in frq])        
                ax2.semilogx(period[1:n//2],abs(y_fft[1:n//2])**2/np.max(abs(y_fft[1:n//2])**2), color = dic_color[year],
                             linewidth=2)#, label='Year = '+str(year))
                ax2.legend(loc='center right', shadow=True,fancybox=True,prop={'size':18})
                #ax2.set_yticks([0, 0.1, 0.2])
                #ax2.set_yticklabels(['0', '0.1', '0.2'])
                plt.text(26, 0.95, 'day', horizontalalignment='left', color='dimgrey', fontsize=14)
                plt.text(24*7+30, 0.95, 'week', horizontalalignment='left', color='dimgrey', fontsize=14)
                plt.text(24*30+100, 0.95, 'month', horizontalalignment='left', color='dimgrey', fontsize=14)
                plt.text(8760+700, 0.95, 'year', horizontalalignment='left', color='dimgrey', fontsize=14)
                if i==0:
                    ax2.set_xticks([1, 10, 100, 1000,8760])
                    ax2.set_xticklabels(['1', '10', '100', '1000','8760'])
                    ax2.set_xlabel('cycling period (hours)',fontsize=18)
                #else: 
                #    ax2.set_xticks([])
                if i==0:
                        ax2.set_title(str(storagetech)+' Fourier Power Series - Distribution cost: ' +str(dist).replace('dist', '')+' (' + str(transition) + ')',fontsize=18)
            
            name = r'\oneplot_'+str(storagetech)+'_'+str(dist).replace('.', '')+ str(transition)
            plt.savefig(pathplot+name, dpi=300, bbox_inches='tight')
            plt.clf()

