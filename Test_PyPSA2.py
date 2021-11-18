# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 12:28:55 2021

@author: Anders
"""




import pypsa
#pandas package is very useful to work with imported data, time series, matrices ...
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
path = r'C:\Users\ander\OneDrive - Aarhus universitet\Maskiningenioer\Kandidat\3. semester\PreProject Master\Network files\Figures testing'

# We start by creating the network. In this example, the country is modelled as a single node, so the network will only include one bus.
# 
# We select the year 2015 and set the hours in that year as snapshots.

# In[2]:
#%%
## Loading the network in a proper way


flex= 'elec_s_37'  
line_limit = 'lv1.0'
co2_limit = 'Co2L0.1'
solar = 'solar+p3-'
cost_dist='1'

network_name= (flex + '_' + line_limit + '__' + co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')

network = pypsa.Network(network_name)         
    

co2_limits=['0.5', '0.2', '0.1', '0.05',  '0'] #, '0.025']
line_limits=['lv1.0','lv1.1','lv1.2','lv1.5','lv2.0']
cost_dists=['0.1','0.5','1','2','10']
d1 = {} # Imported data
d2 = {} # Imported generators
d3 = {} # Specified sizes of generators


for co2_limit in co2_limits:    
    network_name= (flex+ '_' + line_limit + '__' +'Co2L'+co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')
    d1["n"+str(co2_limit)] = pypsa.Network(network_name) 
    d2["generators"+ str(co2_limit)] = d1["n"+str(co2_limit)].generators.groupby("carrier")["p_nom_opt"].sum()
    d3["sizes" + str(co2_limit)] = [d2["generators" +str(co2_limit)]['gas'].sum(),
                       d2["generators"+ str(co2_limit)]['offwind-ac'].sum(),
                       d2["generators"+ str(co2_limit)]['offwind-dc'].sum(),
                       d2["generators"+ str(co2_limit)]['onwind'].sum(),
                       d2["generators"+ str(co2_limit)]['ror'].sum(),
                       d2["generators"+ str(co2_limit)]['solar'].sum(),
                       d2["generators"+ str(co2_limit)]['solar rooftop'].sum()]


# Import all data
flex= 'elec_s_37'  
line_limit = 'lv1.0'
co2_limit = 'Co2L0.1'
solar = 'solar+p3-'
cost_dist='1'

network_name= (flex + '_' + line_limit + '__' + co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')

network = pypsa.Network(network_name)         

co2_limits=['0.5', '0.2', '0.1', '0.05',  '0'] #, '0.025']
line_limits=['lv1.0','lv1.1','lv1.2','lv1.5','lv2.0']
cost_dists=['0.1','0.5','1','2','10']

df = pd.DataFrame()

for cost_dist in cost_dists:
    for line_limit in line_limits:
        for co2_limit in co2_limits:
            network_name= (flex+ '_' + line_limit + '__' +'Co2L'+co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')
            n = pypsa.Network(network_name) 
            generators = n.generators.groupby("carrier")["p_nom_opt"].sum()
            storage = n.storage_units.groupby("carrier")["p_nom_opt"].sum()
            df[cost_dist+line_limit+co2_limit] = generators


#%%
####----- Import data using dataframe instead -----####
####------ PLOT OF MORE FIGURES TOGETHER -----------###

flex= 'elec_s_37'  
lv = 'lv1.0'
co2_limit = 'Co2L0.1'
solar = 'solar+p3-dist'
co2_limits=['Co2L0.5', 'Co2L0.2', 'Co2L0.1', 'Co2L0.05',  'Co2L0'] # the corresponding CO2 limits in the code
lvl = ['1.0', '1.1', '2.0'] #, '1.2', '1.5', '2.0'


df = pd.DataFrame()

for lv in lvl:
    for co2_limit in co2_limits:
        network_name= (flex+ '_' + 'lv'+ lv + '__' +co2_limit+ '-' + solar +'1'+'_'+'2030'+'.nc')
        print(network_name)
        n = pypsa.Network(network_name) 
        generators = n.generators.groupby("carrier")["p_nom_opt"].sum()
        storage = n.storage_units.groupby("carrier")["p_nom_opt"].sum()
        df[lv+co2_limit] = generators



#df.plot.bar(df.filter(regex='1.0'))

df1 = df.filter(like='1.0')
df2 = df.filter(like='1.1')
df3 = df.filter(like='2.0')

#plt.figure()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize = (15, 5))
fig.suptitle('Installed capacity vs. CO2 constrain and Transmission expansion')
df1.plot.bar(ax=ax1,rot=25 )
ax1.set_xlabel('Carrier')
ax1.set_ylabel('Installed capacity [MW]')
ax1.set_title('lv1.0')
ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax1.set_ylim(0,700e3)
ax1.yaxis.grid()
ax1.get_legend().remove()
#ax1.legend(['CO2 50%','CO2 20%','CO2 10%','CO2 5%', 'CO2 0%'])
#plt.rc('grid', linestyle="--", color='gray')
#ax1.legend(frameon = True, ncol = 5, shadow=True, bbox_to_anchor=(0.5, 1.25), loc='upper center', title = '% CO2 emmesion compared to 1990')

df2.plot.bar(ax=ax2,rot=25 )
ax2.set_xlabel('Carrier')
ax2.set_ylabel('Installed capacity [MW]')
ax2.set_title('lv1.1')
ax2.set_ylim(0,700e3)
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#ax2.legend(['CO2 50%','CO2 20%','CO2 10%','CO2 5%', 'CO2 0%'])
ax2.get_legend().remove()
ax2.yaxis.grid()

df3.plot.bar(ax=ax3,rot=25 )
ax3.set_xlabel('Carrier')
ax3.set_ylabel('Installed capacity [MW]')
ax3.set_title('lv2.0')
ax3.set_ylim(0,700e3)
ax3.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax3.legend(['CO2 50%','CO2 20%','CO2 10%','CO2 5%', 'CO2 0%'],bbox_to_anchor=(1.05, 1))
#ax3.get_legend().remove()
ax3.yaxis.grid()

# Save the figure in the selected path
name = r'\subplotsinstalledcapacity.jpg'
plt.show()
plt.savefig(path+name, dpi=300,format='jpg')   


#%%#### Plots
flex= 'elec_s_37'  
lv = 'lv1.0'
co2_limit = 'Co2L0.1'
solar = 'solar+p3-dist'
co2_limits=['Co2L0.5', 'Co2L0.2', 'Co2L0.1', 'Co2L0.05',  'Co2L0'] # the corresponding CO2 limits in the code
lvl = ['1.0', '1.1', '2.0'] #, '1.2', '1.5', '2.0'


df = pd.DataFrame()

for lv in lvl:
    for co2_limit in co2_limits:
        network_name= (flex+ '_' + 'lv'+ lv + '__' +co2_limit+ '-' + solar +'1'+'_'+'2030'+'.nc')
        print(network_name)
        n = pypsa.Network(network_name) 
        generators = n.generators.groupby("carrier")["p_nom_opt"].sum()
        storage = n.storage_units.groupby("carrier")["p_nom_opt"].sum()
        df[lv+co2_limit] = generators

# Links, storage and lines
links=n.links.p_nom_opt
storage=n.storage_units.p_nom_opt
lines=n.lines
stores = n.stores
n.plot()

pypsa.plot.plot(n)
name = r'\linesandlinkes.jpg'
plt.show()
plt.savefig(path+name, dpi=300,format='jpg') 


# Plotting Networks:
import cartopy.crs as ccrs
loading = (n.lines_t.p0.abs().mean().sort_index()/(n.lines.s_nom_opt*n.lines.s_max_pu).sort_index()).fillna(0.)

fig,ax = plt.subplots(
   #figsize(10,10),
    subplot_kw={"projection": ccrs.PlateCarree()}
    )

n.plot(ax=ax,
       bus_colors='gray',
       branch_components=["Line"],
       line_widths=n.lines.s_nom_opt/3e3,
       line_colors=loading,
       line_cmap=plt.cm.viridis,
       color_geomap=True,
       bus_sizes=0)
ax.axis('off');
name = r'\linesforeurope.jpg'
plt.show()
plt.savefig(path+name, dpi=300,format='jpg') 


#%%
## PLOT OF CO2 limits for line limit 1.0 ##
fig = plt.figure(figsize = (10, 5))
X = np.arange(7)  # Antallet af generatorer

plt.bar(X,d3["sizes0"],0.15,label='CO$_2$=0')
plt.bar(X+0.15,d3["sizes0.05"],0.15,label='CO$_2$=0.05')   
plt.bar(X+0.3,d3["sizes0.1"],0.15,label='CO$_2$=0.1')
plt.bar(X+0.45,d3["sizes0.2"],0.15,label='CO$_2$=0.2') 
plt.bar(X+0.60,d3["sizes0.5"],0.15,label='CO$_2$=0.5')      
plt.legend()

plt.xticks([i + 0.25 for i in range(7)], ['gas',
          'offwind-ac',
          'offwind-dc',
          'onwind',
          'ror',
          'solar', 
          'solarrooftop'])

plt.title("Bar plot representing the effects of the CO$_2$ constraint for the EU (linelimit lv1.0, costdist 0.1)")
#plt.xlabel('Generators')
plt.ylabel('Capacity of generators (MW)')
plt.show()
plt.savefig(r'C:\Users\ander\OneDrive - Aarhus universitet\Maskiningenioer\Kandidat\3. semester\PreProject Master\Network files\Figures testing\co2constraintEurope.jpg', format='jpg', dpi=300)


#%% Investigation af CO_2 price

flex= 'elec_s_37'  
line_limit = 'lv1.0'
co2_limit = 'Co2L0.1'
solar = 'solar+p3-'
cost_dist='1'

network_name= (flex + '_' + line_limit + '__' + co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')

network = pypsa.Network(network_name)         

co2_limits=['0.5', '0.2', '0.1', '0.05',  '0'] #, '0.025']
line_limit='lv1.0'
cost_dists=['0.1','0.5','1','2','10']

df = pd.DataFrame()
dfcon = pd.DataFrame()
dftot = pd.Series()

for cost_dist in cost_dists:
    for co2_limit in co2_limits:
        network_name= (flex+ '_' + line_limit + '__' +'Co2L'+co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')
        n = pypsa.Network(network_name) 
        constraint = n.global_constraints.constant
        price = n.global_constraints.mu
        totalcost=n.objective/1000000
        df[cost_dist+line_limit+co2_limit] = price
        dfcon[cost_dist+line_limit+co2_limit] = constraint
        dftot[cost_dist+line_limit+co2_limit]=totalcost
    

print(network.global_constraints.constant) #CO2 limit (constant in the constraint)

print(network.global_constraints.mu) #CO2 price (Lagrance multiplier in the constraint)

df=df.dropna()

#dftotcost=pd.Series()

dftotlist=list(dftot)

n=0
x=[50, 20, 10, 5,0]
# Plot of different total system costs:
while n<25:
    if n<4:
        name='Cost dist 0.1'
    if n>4:
        name='Cost dist 0.5'
    if n>9:
        name='Cost dist 1'
    if n>14:
        name='Cost dist 2'
    if n>19:
        name='Cost dist 10'
    plt.plot(x,dftotlist[n:5+n],label=str(name))
    n=n+5

plt.legend()
plt.xlabel("CO$_2$ limit (% reduction compared to 1990)")
plt.ylabel("Total costs of system [million Euros]")
plt.legend(loc='upper left')
plt.title('Total costs of system')
plt.gca().invert_xaxis()
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.grid()
plt.show()
name = r'\totalsystemcost.jpg'
plt.savefig(r'path+name', format='jpg', dpi=300)
    



# Plot of CO2 price
ax = df.plot.bar()
ax.set_xlabel('CO2 limit')
ax.set_ylabel('CO2 price [Euro/tons]')
ax.set_title('CO2 price for Europe at different \n CO2 constraints (Line limit=1, cost_dist=1)')
ax.set_yscale('log')
#plt.xticks(rotation=90)
#plt.xticks(rotation=180, ha='right')
plt.tick_params( # Removes x-ticks
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off 
plt.show()
name = r'\co2price.jpg'
plt.savefig(path+name, dpi=300,format='jpg' ,bbox_inches='tight') 

## Plot of total system costs ##





#%%
## PLOT OF CO2 limits effect on optimal rooftop solar vs. groundmounted  ##
fig = plt.figure(figsize = (10, 5))
X = np.arange(2)  # Antallet af generatorer

plt.bar(X,d3["sizes0"][5:7],0.15,label='CO$_2$=0')
plt.bar(X+0.15,d3["sizes0.05"][5:7],0.15,label='CO$_2$=0.05')   
plt.bar(X+0.3,d3["sizes0.1"][5:7],0.15,label='CO$_2$=0.1')
plt.bar(X+0.45,d3["sizes0.2"][5:7],0.15,label='CO$_2$=0.2') 
plt.bar(X+0.60,d3["sizes0.5"][5:7],0.15,label='CO$_2$=0.5')      
plt.legend()

plt.xticks([i + 0.25 for i in range(2)], ['solar', 
          'solarrooftop'])

plt.title("Bar plot representing the effects of the CO$_2$ constraint for the EU (linelimit lv1.0, costdist 0.1)")
#plt.xlabel('Generators')
plt.ylabel('Capacity of generators (MW)')
plt.show()
plt.savefig(r'C:\Users\ander\OneDrive - Aarhus universitet\Maskiningenioer\Kandidat\3. semester\PreProject Master\Network files\Figures testing\rooftopandgroundsolarEurope.jpg', format='jpg', dpi=300)


#%%
## PLOT OF CO2 limits effect on optimal rooftop solar vs. groundmounted FOR SPAIN  ##

# Import data for Spain:
co2_limits=['0.5', '0.2', '0.1', '0.05',  '0'] #, '0.025']
line_limits='lv1.0' #['lv1.0','lv1.1','lv1.2','lv1.5','lv2.0']
cost_dists='0.1' #['0.1','0.5','1','2','10']
d1spain = {} # Imported data
d2spain1 = {} # Imported generators
d2spain2 = {} # Imported generators
d3spain = {} # Specified sizes of generators


for co2_limit in co2_limits:    
    network_name= (flex+ '_' + line_limit + '__' +'Co2L'+co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')
    d1spain["n"+str(co2_limit)] = pypsa.Network(network_name) 
    d2spain1["ESsolar"+ str(co2_limit)] = d1spain["n"+str(co2_limit)].generators.p_nom_opt["AT0 0 solar"]
    d2spain2["ESsolarooftop"+ str(co2_limit)] = d1spain["n"+str(co2_limit)].generators.p_nom_opt["ES0 0 solar rooftop"]

d2spain1.update(d2spain2)

d2spainlist=list(d2spain1.items())
d2spainvalues=list(d2spain1.values())

fig = plt.figure(figsize = (10, 5))
X = np.arange(2)  # Antallet af generatorer


plt.bar(X,[d2spainvalues[4],d2spainvalues[9]],0.15,label='CO$_2$=0')
plt.bar(X+0.15,[d2spainvalues[3],d2spainvalues[8]],0.15,label='CO$_2$=0.05')   
plt.bar(X+0.3,[d2spainvalues[2],d2spainvalues[7]],0.15,label='CO$_2$=0.1')
plt.bar(X+0.45,[d2spainvalues[1],d2spainvalues[6]],0.15,label='CO$_2$=0.2') 
plt.bar(X+0.60,[d2spainvalues[0],d2spainvalues[5]],0.15,label='CO$_2$=0.5')      
plt.legend()

plt.xticks([i + 0.25 for i in range(2)], ['solar', 
          'solarrooftop'])

plt.title("Bar plot representing the effects of the CO$_2$ constraint for the solar rooftop \n and solar for Spain (linelimit lv1.0, costdist 0.1)")
#plt.xlabel('Generators')
plt.ylabel('Capacity of solar and solar rooftop (MW)')
plt.show()
plt.savefig(r'C:\Users\ander\OneDrive - Aarhus Universitet\Maskiningenioer\Kandidat\3. semester\PreProject Master\Network files\Figures testing\rooftopandgroundsolarSpain.jpg', format='jpg', dpi=300)


#%%
## OPTIMAL CAPACITIES FOR ALL OF EUROPE ##
flex= 'elec_s_37'  
line_limit = 'lv1.0'
co2_limit = 'Co2L0.1'
solar = 'solar+p3-'
cost_dist='1'

network_name= (flex + '_' + line_limit + '__' + co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')

network = pypsa.Network(network_name)         

co2_limits=['0.5', '0.2', '0.1', '0.05',  '0'] #, '0.025']
line_limits='lv1.0' #['lv1.0','lv1.1','lv1.2','lv1.5','lv2.0']
cost_dists='0.1' #['0.1','0.5','1','2','10']
d1 = {} # Imported data

for co2_limit in co2_limits:    
    network_name= (flex+ '_' + line_limit + '__' +'Co2L'+co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')
    d1["n"+str(co2_limit)] = pypsa.Network(network_name) 

d1["n0"].plot()


import cartopy.crs as ccrs
loading = (d1["n0"].lines_t.p0.abs().mean().sort_index()/(d1["n0"].lines.s_nom_opt*d1["n0"].lines.s_max_pu).sort_index()).fillna(0.)

fig,ax = plt.subplots(
   #figsize(10,10),
    subplot_kw={"projection": ccrs.PlateCarree()}
    )

d1["n0"].plot(ax=ax,
       bus_colors='gray',
       branch_components=["Line"],
       line_widths=d1["n0"].lines.s_nom_opt/3e3,
       line_colors=loading,
       line_cmap=plt.cm.viridis,
       color_geomap=True,
       bus_sizes=0)
ax.axis('off');

#%% Duration curves


d1["n0"].loads_t.p_set.sum(axis=1).plot(figsize=(15,3)) # Plot for all generators (duration)


d1["n0"].generators_t.p_max_pu.loc["ES0 0 solar"].plot(figsize=(15.3))




#%% Plot of Fourier Power Series
pathdata = r'C:\Users\ander\OneDrive - Aarhus universitet\Maskiningenioer\Kandidat\3. semester\PreProject Master\Network files'


flex= 'elec_s_37'  
line_limit = 'lv1.0'
co2_limit = '0.1'
solar = 'solar+p3-'
cost_dist='1'

#line_limit='0.125' 
co2_limits=['0.5', '0.2', '0.1', '0.05',  '0']

flexs = ['elec_s_37'] 
techs=['PHS', 'ror']

network_name= (flex + '_' + line_limit + '__' +'Co2L'+ co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')

network = pypsa.Network(network_name)         

datos = pd.DataFrame(index=pd.MultiIndex.from_product([pd.Series(data=techs, name='tech',),
                                                       pd.Series(data=flexs, name='flex',),
                                                       pd.Series(data=co2_limits, name='co2_limits',)]), 
                      columns=pd.Series(data=np.arange(0,8760), name='hour',))
idx = pd.IndexSlice

for co2_limit in co2_limits:
        network_name= (flex + '_' + line_limit + '__' +'Co2L'+ co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')  
        network = pypsa.Network(network_name)
        datos.loc[idx['PHS', flex ,co2_limit], :] = np.array(network.storage_units_t.state_of_charge[network.storage_units.index[network.storage_units.carrier == 'PHS']].sum(axis=1)/(6*network.storage_units.p_nom[network.storage_units.index[network.storage_units.carrier == 'PHS']].sum()))
        datos.loc[idx['ror', flex, co2_limit], :] = np.array(network.stores_t.e[network.stores.index[network.stores.index.str[3:] == 'ror']].sum(axis=1)/network.stores.e_nom_opt[network.stores.index[network.stores.index.str[3:] == 'ror']].sum())

# Save dataframe to pickled pandas object and csv file
datos.to_pickle(pathdata+'\data_for_figures/storage_timeseries.pickle') 
datos.to_csv(pathdata+'\data_for_figures/storage_timeseries.csv', sep=',') 


## The plot
##### Figure of the Fourier transform for the PHS charging patterns
datos=pd.read_csv(pathdata+'\data_for_figures/storage_timeseries.csv', sep=',', header=0, index_col=(0,1,2))


plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.figure(figsize=(10, 10))
gs1 = gridspec.GridSpec(10, 1)
gs1.update(wspace=0.05)

ax1 = plt.subplot(gs1[0:3,0])
ax1.set_ylabel('PHS filling level')
ax1.set_xlabel('hour')
ax1.set_xlim(0,8760)
ax1.set_ylim(0,1)

flex='elec_s_37'#'elec_central' #'elec_only'

co2_limits=['0.5', '0.2', '0']
storage_names=['PHS'] #,'battery','H2']
dic_color={'PHS':'darkgreen'}
storage_names=['PHS'] #,'battery','H2']
dic_color={'0.5':'olive','0.2':'darkgreen','0':'red'}
dic_label={'0.5':'50%','0.2':'20%','0':'0%'}
dic_alpha={'0.5':1,'0.2':1,'0':1}
dic_linewidth={'0.5':2,'0.2':2,'0':2}

for i,co2_lim in enumerate(co2_limits):
    ax2 = plt.subplot(gs1[4+2*i:6+2*i,0])    
    ax2.set_xlim(1,10000)
    ax2.set_ylim(0,1.2)
    plt.axvline(x=24, color='lightgrey', linestyle='--')
    plt.axvline(x=24*7, color='lightgrey', linestyle='--')
    plt.axvline(x=24*30, color='lightgrey', linestyle='--')
    plt.axvline(x=8760, color='lightgrey', linestyle='--')   
    ax1.plot(np.arange(0,8760), datos.loc[idx['PHS', flex, float(co2_lim)], :]/np.max(datos.loc[idx['PHS', flex, float(co2_lim)], :]), 
             color=dic_color[co2_lim], alpha=dic_alpha[co2_lim], linewidth=dic_linewidth[co2_lim],
             label='CO$_2$='+dic_label[co2_lim])
    ax1.legend(loc=(0.2, 1.05), ncol=3, shadow=True,fancybox=True,prop={'size':18})
    n_years=1
    t_sampling=1 # sampling rate, 1 data per hour
    x = np.arange(1,8761*n_years, t_sampling) 
    y = np.hstack([np.array(datos.loc[idx['PHS', flex, float(co2_lim)], :])]*n_years)
    n = len(x)
    y_fft=np.fft.fft(y)/n #n for normalization    
    frq=np.arange(0,1/t_sampling,1/(t_sampling*n))        
    period=np.array([1/f for f in frq])        
    ax2.semilogx(period[1:n//2],abs(y_fft[1:n//2])**2/np.max(abs(y_fft[1:n//2])**2), color=dic_color[co2_lim],
                 linewidth=2, label='CO$_2$ = '+dic_label[co2_lim])  
    ax2.legend(loc='center right', shadow=True,fancybox=True,prop={'size':18})
    #ax2.set_yticks([0, 0.1, 0.2])
    #ax2.set_yticklabels(['0', '0.1', '0.2'])
    plt.text(26, 0.95, 'day', horizontalalignment='left', color='dimgrey', fontsize=14)
    plt.text(24*7+20, 0.95, 'week', horizontalalignment='left', color='dimgrey', fontsize=14)
    plt.text(24*30+20, 0.95, 'month', horizontalalignment='left', color='dimgrey', fontsize=14)
    if i==2:
        ax2.set_xticks([1, 10, 100, 1000, 10000])
        ax2.set_xticklabels(['1', '10', '100', '1000', '10000'])
        ax2.set_xlabel('cycling period (hours)')
    else: 
        ax2.set_xticks([])
name = r'\Fourier_transform_PHS.jpg'
plt.show()
plt.savefig(path+name,dpi=300,format='jpg') #bbox_inches='tight' 

## Plot for run of river
plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.figure(figsize=(10, 5))
gs1 = gridspec.GridSpec(7, 1)
gs1.update(wspace=0.05)

co2_limits=['0.5', '0.2', '0']
storage_names=['ror'] #,'battery','H2']
dic_color={'ror':'darkgreen'}
storage_names=['ror'] #,'battery','H2']
dic_color={'0.5':'olive','0.2':'darkgreen','0':'red'}
dic_label={'0.5':'50%','0.2':'20%','0':'0%'}
dic_alpha={'0.5':1,'0.2':1,'0':1}
dic_linewidth={'0.5':2,'0.2':2,'0':2}

for i,co2_lim in enumerate(co2_limits):
    ax2 = plt.subplot(gs1[0+2*i:2+2*i,0])    #[4+2*i:6+2*i,0] 
    ax2.set_xlim(1,10000)
    ax2.set_ylim(0,1.2)
    plt.axvline(x=24, color='lightgrey', linestyle='--')
    plt.axvline(x=24*7, color='lightgrey', linestyle='--')
    plt.axvline(x=24*30, color='lightgrey', linestyle='--')
    plt.axvline(x=8760, color='lightgrey', linestyle='--')   
    #ax1.plot(np.arange(0,8760), datos.loc[idx['PHS', flex, float(co2_lim)], :]/np.max(datos.loc[idx['PHS', flex, float(co2_lim)], :]), 
    #         color=dic_color[co2_lim], alpha=dic_alpha[co2_lim], linewidth=dic_linewidth[co2_lim],
    #         label='CO$_2$='+dic_label[co2_lim])
    #ax1.legend(loc=(0.2, 1.05), ncol=3, shadow=True,fancybox=True,prop={'size':18})
    n_years=1
    t_sampling=1 # sampling rate, 1 data per hour
    x = np.arange(1,8761*n_years, t_sampling) 
    y = np.hstack([np.array(datos.loc[idx['PHS', flex, float(co2_lim)], :])]*n_years)
    n = len(x)
    y_fft=np.fft.fft(y)/n #n for normalization    
    frq=np.arange(0,1/t_sampling,1/(t_sampling*n))        
    period=np.array([1/f for f in frq])        
    ax2.semilogx(period[1:n//2],abs(y_fft[1:n//2])**2/np.max(abs(y_fft[1:n//2])**2), color=dic_color[co2_lim],
                 linewidth=2, label='CO$_2$ = '+dic_label[co2_lim])  
    ax2.legend(loc='center right', shadow=True,fancybox=True,prop={'size':18})
    #ax2.set_yticks([0, 0.1, 0.2])
    #ax2.set_yticklabels(['0', '0.1', '0.2'])
    plt.text(26, 0.95, 'day', horizontalalignment='left', color='dimgrey', fontsize=14)
    plt.text(24*7+20, 0.95, 'week', horizontalalignment='left', color='dimgrey', fontsize=14)
    plt.text(24*30+20, 0.95, 'month', horizontalalignment='left', color='dimgrey', fontsize=14)
    if i==2:
        ax2.set_xticks([1, 10, 100, 1000, 10000])
        ax2.set_xticklabels(['1', '10', '100', '1000', '10000'])
        ax2.set_xlabel('cycling period (hours)')
    else: 
        ax2.set_xticks([])
#plt.title('Run of River Fourier power spectrum')
name = r'\Fourier_transform_ror.jpg'
plt.show()
plt.savefig(path+name,dpi=300,format='jpg')

#%% Fourier Power Series for Solar

flex= 'elec_s_37'  
line_limit = 'lv1.0'
co2_limit = '0.1'
solar = 'solar+p3-'
cost_dist='1'

#line_limit='0.125' 
co2_limits=['0.5', '0.2', '0.1', '0.05',  '0']

flexs = ['elec_s_37'] 
techs=['solar']

network_name= (flex + '_' + line_limit + '__' +'Co2L'+ co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')

network = pypsa.Network(network_name)         

datos = pd.DataFrame(index=pd.MultiIndex.from_product([pd.Series(data=techs, name='tech',),
                                                       pd.Series(data=flexs, name='flex',),
                                                       pd.Series(data=co2_limits, name='co2_limits',)]), 
                      columns=pd.Series(data=np.arange(0,8760), name='hour',))
idx = pd.IndexSlice

for co2_limit in co2_limits:
        network_name= (flex + '_' + line_limit + '__' +'Co2L'+ co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')  
        network = pypsa.Network(network_name)
        datos.loc[idx['solar', flex ,co2_limit], :] = np.array(network.generators_t.p["ES0 0 solar"])

# Save dataframe to pickled pandas object and csv file
datos.to_pickle(pathdata+'\data_for_figures/solar_timeseries.pickle') 
datos.to_csv(pathdata+'\data_for_figures/solar_timeseries.csv', sep=',') 


## The plot
##### Figure of the Fourier transform for the PHS charging patterns
datos=pd.read_csv(pathdata+'\data_for_figures/solar_timeseries.csv', sep=',', header=0, index_col=(0,1,2))


plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.figure(figsize=(10, 10))
gs1 = gridspec.GridSpec(10, 1)
gs1.update(wspace=0.05)

ax1 = plt.subplot(gs1[0:3,0])
ax1.set_ylabel('Solar capacity factor')
ax1.set_xlabel('hour')
ax1.set_xlim(0,8760)
ax1.set_ylim(0,1)

flex='elec_s_37'#'elec_central' #'elec_only'

co2_limits=['0.5', '0.2', '0']
storage_names=['solar'] #,'battery','H2']
dic_color={'solar':'darkgreen'}
storage_names=['solar'] #,'battery','H2']
dic_color={'0.5':'olive','0.2':'darkgreen','0':'red'}
dic_label={'0.5':'50%','0.2':'20%','0':'0%'}
dic_alpha={'0.5':1,'0.2':1,'0':1}
dic_linewidth={'0.5':2,'0.2':2,'0':2}

for i,co2_lim in enumerate(co2_limits):
    ax2 = plt.subplot(gs1[4+2*i:6+2*i,0])    
    ax2.set_xlim(1,10000)
    ax2.set_ylim(0,1.2)
    plt.axvline(x=24, color='lightgrey', linestyle='--')
    plt.axvline(x=24*7, color='lightgrey', linestyle='--')
    plt.axvline(x=24*30, color='lightgrey', linestyle='--')
    plt.axvline(x=8760, color='lightgrey', linestyle='--')   
    ax1.plot(np.arange(0,8760), datos.loc[idx['solar', flex, float(co2_lim)], :]/np.max(datos.loc[idx['solar', flex, float(co2_lim)], :]), 
             color=dic_color[co2_lim], alpha=dic_alpha[co2_lim], linewidth=dic_linewidth[co2_lim],
             label='CO$_2$='+dic_label[co2_lim])
    ax1.legend(loc=(0.2, 1.05), ncol=3, shadow=True,fancybox=True,prop={'size':18})
    n_years=1
    t_sampling=1 # sampling rate, 1 data per hour
    x = np.arange(1,8761*n_years, t_sampling) 
    y = np.hstack([np.array(datos.loc[idx['solar', flex, float(co2_lim)], :])]*n_years)
    n = len(x)
    y_fft=np.fft.fft(y)/n #n for normalization    
    frq=np.arange(0,1/t_sampling,1/(t_sampling*n))        
    period=np.array([1/f for f in frq])        
    ax2.semilogx(period[1:n//2],abs(y_fft[1:n//2])**2/np.max(abs(y_fft[1:n//2])**2), color=dic_color[co2_lim],
                 linewidth=2, label='CO$_2$ = '+dic_label[co2_lim])  
    ax2.legend(loc='center right', shadow=True,fancybox=True,prop={'size':18})
    #ax2.set_yticks([0, 0.1, 0.2])
    #ax2.set_yticklabels(['0', '0.1', '0.2'])
    plt.text(26, 0.95, 'day', horizontalalignment='left', color='dimgrey', fontsize=14)
    plt.text(24*7+20, 0.95, 'week', horizontalalignment='left', color='dimgrey', fontsize=14)
    plt.text(24*30+20, 0.95, 'month', horizontalalignment='left', color='dimgrey', fontsize=14)
    if i==2:
        ax2.set_xticks([1, 10, 100, 1000, 10000])
        ax2.set_xticklabels(['1', '10', '100', '1000', '10000'])
        ax2.set_xlabel('cycling period (hours)')
    else: 
        ax2.set_xticks([])
name = r'\Fourier_transform_solar.jpg'
plt.show()
plt.savefig(path+name,dpi=300,format='jpg') #bbox_inches='tight' 




#%% Fourier Power Series for Wind

flex= 'elec_s_37'  
line_limit = 'lv1.0'
co2_limit = '0.1'
solar = 'solar+p3-'
cost_dist='1'

#line_limit='0.125' 
co2_limits=['0.5', '0.2', '0.1', '0.05',  '0']

flexs = ['elec_s_37'] 
techs=['onwind']

network_name= (flex + '_' + line_limit + '__' +'Co2L'+ co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')

network = pypsa.Network(network_name)         

datos = pd.DataFrame(index=pd.MultiIndex.from_product([pd.Series(data=techs, name='tech',),
                                                       pd.Series(data=flexs, name='flex',),
                                                       pd.Series(data=co2_limits, name='co2_limits',)]), 
                      columns=pd.Series(data=np.arange(0,8760), name='hour',))
idx = pd.IndexSlice

for co2_limit in co2_limits:
        network_name= (flex + '_' + line_limit + '__' +'Co2L'+ co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')  
        network = pypsa.Network(network_name)
        datos.loc[idx['onwind', flex ,co2_limit], :] = np.array(network.generators_t.p["DK0 0 onwind"])

# Save dataframe to pickled pandas object and csv file
datos.to_pickle(pathdata+'\data_for_figures/onwind_timeseries.pickle') 
datos.to_csv(pathdata+'\data_for_figures/onwind_timeseries.csv', sep=',') 


## The plot
##### Figure of the Fourier transform for the PHS charging patterns
datos=pd.read_csv(pathdata+'\data_for_figures/onwind_timeseries.csv', sep=',', header=0, index_col=(0,1,2))


plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.figure(figsize=(10, 10))
gs1 = gridspec.GridSpec(10, 1)
gs1.update(wspace=0.05)

ax1 = plt.subplot(gs1[0:3,0])
ax1.set_ylabel('Onwind capacity factor')
ax1.set_xlabel('hour')
ax1.set_xlim(0,8760)
ax1.set_ylim(0,1)

flex='elec_s_37'#'elec_central' #'elec_only'

co2_limits=['0.5', '0.2', '0']
storage_names=['onwind'] #,'battery','H2']
dic_color={'onwind':'darkgreen'}
storage_names=['onwind'] #,'battery','H2']
dic_color={'0.5':'olive','0.2':'darkgreen','0':'red'}
dic_label={'0.5':'50%','0.2':'20%','0':'0%'}
dic_alpha={'0.5':1,'0.2':1,'0':1}
dic_linewidth={'0.5':2,'0.2':2,'0':2}

for i,co2_lim in enumerate(co2_limits):
    ax2 = plt.subplot(gs1[4+2*i:6+2*i,0])    
    ax2.set_xlim(1,10000)
    ax2.set_ylim(0,1.2)
    plt.axvline(x=24, color='lightgrey', linestyle='--')
    plt.axvline(x=24*7, color='lightgrey', linestyle='--')
    plt.axvline(x=24*30, color='lightgrey', linestyle='--')
    plt.axvline(x=8760, color='lightgrey', linestyle='--')   
    ax1.plot(np.arange(0,8760), datos.loc[idx['onwind', flex, float(co2_lim)], :]/np.max(datos.loc[idx['onwind', flex, float(co2_lim)], :]), 
             color=dic_color[co2_lim], alpha=dic_alpha[co2_lim], linewidth=dic_linewidth[co2_lim],
             label='CO$_2$='+dic_label[co2_lim])
    ax1.legend(loc=(0.2, 1.05), ncol=3, shadow=True,fancybox=True,prop={'size':18})
    n_years=1
    t_sampling=1 # sampling rate, 1 data per hour
    x = np.arange(1,8761*n_years, t_sampling) 
    y = np.hstack([np.array(datos.loc[idx['onwind', flex, float(co2_lim)], :])]*n_years)
    n = len(x)
    y_fft=np.fft.fft(y)/n #n for normalization    
    frq=np.arange(0,1/t_sampling,1/(t_sampling*n))        
    period=np.array([1/f for f in frq])        
    ax2.semilogx(period[1:n//2],abs(y_fft[1:n//2])**2/np.max(abs(y_fft[1:n//2])**2), color=dic_color[co2_lim],
                 linewidth=2, label='CO$_2$ = '+dic_label[co2_lim])  
    ax2.legend(loc='center right', shadow=True,fancybox=True,prop={'size':18})
    #ax2.set_yticks([0, 0.1, 0.2])
    #ax2.set_yticklabels(['0', '0.1', '0.2'])
    plt.text(26, 0.95, 'day', horizontalalignment='left', color='dimgrey', fontsize=14)
    plt.text(24*7+20, 0.95, 'week', horizontalalignment='left', color='dimgrey', fontsize=14)
    plt.text(24*30+20, 0.95, 'month', horizontalalignment='left', color='dimgrey', fontsize=14)
    if i==2:
        ax2.set_xticks([1, 10, 100, 1000, 10000])
        ax2.set_xticklabels(['1', '10', '100', '1000', '10000'])
        ax2.set_xlabel('cycling period (hours)')
    else: 
        ax2.set_xticks([])
name = r'\Fourier_transform_onwind.jpg'
plt.show()
plt.savefig(path+name,dpi=300,format='jpg') #bbox_inches='tight' 


#%% Fourier Power Series for Wind (Summed)

flex= 'elec_s_37'  
line_limit = 'lv1.0'
co2_limit = '0.1'
solar = 'solar+p3-'
cost_dist='1'

#line_limit='0.125' 
co2_limits=['0.5', '0.2', '0.1', '0.05',  '0']

flexs = ['elec_s_37'] 
techs=['onwind']

network_name= (flex + '_' + line_limit + '__' +'Co2L'+ co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')

network = pypsa.Network(network_name)         

datos = pd.DataFrame(index=pd.MultiIndex.from_product([pd.Series(data=techs, name='tech',),
                                                       pd.Series(data=flexs, name='flex',),
                                                       pd.Series(data=co2_limits, name='co2_limits',)]), 
                      columns=pd.Series(data=np.arange(0,8760), name='hour',))
idx = pd.IndexSlice
df = pd.DataFrame()
generators=pd.DataFrame()

for co2_limit in co2_limits:
        network_name= (flex + '_' + line_limit + '__' +'Co2L'+ co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')  
        network = pypsa.Network(network_name)
        generators = np.array(network.generators_t.p[network.generators.index[network.generators.carrier == 'onwind']].sum(axis=1))
        datos.loc[idx['onwind', flex ,co2_limit], :] = generators

# Save dataframe to pickled pandas object and csv file
datos.to_pickle(pathdata+'\data_for_figures/onwind_summed_timeseries.pickle') 
datos.to_csv(pathdata+'\data_for_figures/onwind_summed_timeseries.csv', sep=',') 


## The plot
##### Figure of the Fourier transform for the PHS charging patterns
datos=pd.read_csv(pathdata+'\data_for_figures/onwind_summed_timeseries.csv', sep=',', header=0, index_col=(0,1,2))


plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.figure(figsize=(10, 10))
gs1 = gridspec.GridSpec(10, 1)
gs1.update(wspace=0.05)

ax1 = plt.subplot(gs1[0:3,0])
ax1.set_ylabel('Onwind capacity factor')
ax1.set_xlabel('hour')
ax1.set_xlim(0,8760)
ax1.set_ylim(0,1)

flex='elec_s_37'#'elec_central' #'elec_only'

co2_limits=['0.5', '0.2', '0']
storage_names=['onwind'] #,'battery','H2']
dic_color={'onwind':'darkgreen'}
storage_names=['onwind'] #,'battery','H2']
dic_color={'0.5':'olive','0.2':'darkgreen','0':'red'}
dic_label={'0.5':'50%','0.2':'20%','0':'0%'}
dic_alpha={'0.5':1,'0.2':1,'0':1}
dic_linewidth={'0.5':2,'0.2':2,'0':2}

for i,co2_lim in enumerate(co2_limits):
    ax2 = plt.subplot(gs1[4+2*i:6+2*i,0])    
    ax2.set_xlim(1,10000)
    ax2.set_ylim(0,1.2)
    plt.axvline(x=24, color='lightgrey', linestyle='--')
    plt.axvline(x=24*7, color='lightgrey', linestyle='--')
    plt.axvline(x=24*30, color='lightgrey', linestyle='--')
    plt.axvline(x=8760, color='lightgrey', linestyle='--')   
    ax1.plot(np.arange(0,8760), datos.loc[idx['onwind', flex, float(co2_lim)], :]/np.max(datos.loc[idx['onwind', flex, float(co2_lim)], :]), 
             color=dic_color[co2_lim], alpha=dic_alpha[co2_lim], linewidth=dic_linewidth[co2_lim],
             label='CO$_2$='+dic_label[co2_lim])
    ax1.legend(loc=(0.2, 1.05), ncol=3, shadow=True,fancybox=True,prop={'size':18})
    n_years=1
    t_sampling=1 # sampling rate, 1 data per hour
    x = np.arange(1,8761*n_years, t_sampling) 
    y = np.hstack([np.array(datos.loc[idx['onwind', flex, float(co2_lim)], :])]*n_years)
    n = len(x)
    y_fft=np.fft.fft(y)/n #n for normalization    
    frq=np.arange(0,1/t_sampling,1/(t_sampling*n))        
    period=np.array([1/f for f in frq])        
    ax2.semilogx(period[1:n//2],abs(y_fft[1:n//2])**2/np.max(abs(y_fft[1:n//2])**2), color=dic_color[co2_lim],
                 linewidth=2, label='CO$_2$ = '+dic_label[co2_lim])  
    ax2.legend(loc='center right', shadow=True,fancybox=True,prop={'size':18})
    #ax2.set_yticks([0, 0.1, 0.2])
    #ax2.set_yticklabels(['0', '0.1', '0.2'])
    plt.text(26, 0.95, 'day', horizontalalignment='left', color='dimgrey', fontsize=14)
    plt.text(24*7+20, 0.95, 'week', horizontalalignment='left', color='dimgrey', fontsize=14)
    plt.text(24*30+20, 0.95, 'month', horizontalalignment='left', color='dimgrey', fontsize=14)
    if i==2:
        ax2.set_xticks([1, 10, 100, 1000, 10000])
        ax2.set_xticklabels(['1', '10', '100', '1000', '10000'])
        ax2.set_xlabel('cycling period (hours)')
    else: 
        ax2.set_xticks([])
name = r'\Fourier_transform_onwind_summed.jpg'
plt.show()
plt.savefig(path+name,dpi=300,format='jpg') #bbox_inches='tight' 


#%% Fourier Transform on Hydrogen storage (summed)
pathdata = r'C:\Users\ander\OneDrive - Aarhus universitet\Maskiningenioer\Kandidat\3. semester\PreProject Master\Network files'

flex= 'elec_s_37'  
line_limit = 'lv1.0'
co2_limit = '0.1'
solar = 'solar+p3-'
cost_dist='1'

#line_limit='0.125' 
co2_limits=['0.5', '0.2', '0.1', '0.05',  '0']

flexs = ['elec_s_37'] 
techs=['H2']

network_name= (flex + '_' + line_limit + '__' +'Co2L'+ co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')

network = pypsa.Network(network_name)         

datos = pd.DataFrame(index=pd.MultiIndex.from_product([pd.Series(data=techs, name='tech',),
                                                       pd.Series(data=flexs, name='flex',),
                                                       pd.Series(data=co2_limits, name='co2_limits',)]), 
                      columns=pd.Series(data=np.arange(0,8760), name='hour',))
idx = pd.IndexSlice
df = pd.DataFrame()
generators=pd.DataFrame()

for co2_limit in co2_limits:
        network_name= (flex + '_' + line_limit + '__' +'Co2L'+ co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')  
        network = pypsa.Network(network_name)
        generators = np.array(network.stores_t.p[network.stores.index[network.stores.carrier == 'H2']].sum(axis=1))
        datos.loc[idx['H2', flex ,co2_limit], :] = generators

# Save dataframe to pickled pandas object and csv file
datos.to_pickle(pathdata+'\data_for_figures/H2_summed_timeseries.pickle') 
datos.to_csv(pathdata+'\data_for_figures/H2_summed_timeseries.csv', sep=',') 


## The plot
##### Figure of the Fourier transform for the PHS charging patterns
datos=pd.read_csv(pathdata+'\data_for_figures/H2_summed_timeseries.csv', sep=',', header=0, index_col=(0,1,2))


plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.figure(figsize=(10, 10))
gs1 = gridspec.GridSpec(10, 1)
gs1.update(wspace=0.05)

ax1 = plt.subplot(gs1[0:3,0])
ax1.set_ylabel('H2 filling level')
ax1.set_xlabel('hour')
ax1.set_xlim(0,8760)
ax1.set_ylim(0,1)

flex='elec_s_37'#'elec_central' #'elec_only'

co2_limits=['0.5', '0.2', '0']
storage_names=['H2'] #,'battery','H2']
dic_color={'H2':'darkgreen'}
storage_names=['H2'] #,'battery','H2']
dic_color={'0.5':'olive','0.2':'darkgreen','0':'red'}
dic_label={'0.5':'50%','0.2':'20%','0':'0%'}
dic_alpha={'0.5':1,'0.2':1,'0':1}
dic_linewidth={'0.5':2,'0.2':2,'0':2}

for i,co2_lim in enumerate(co2_limits):
    ax2 = plt.subplot(gs1[4+2*i:6+2*i,0])    
    ax2.set_xlim(1,10000)
    ax2.set_ylim(0,1.2)
    plt.axvline(x=24, color='lightgrey', linestyle='--')
    plt.axvline(x=24*7, color='lightgrey', linestyle='--')
    plt.axvline(x=24*30, color='lightgrey', linestyle='--')
    plt.axvline(x=8760, color='lightgrey', linestyle='--')   
    ax1.plot(np.arange(0,8760), datos.loc[idx['H2', flex, float(co2_lim)], :]/np.max(datos.loc[idx['H2', flex, float(co2_lim)], :]), 
             color=dic_color[co2_lim], alpha=dic_alpha[co2_lim], linewidth=dic_linewidth[co2_lim],
             label='CO$_2$='+dic_label[co2_lim])
    ax1.legend(loc=(0.2, 1.05), ncol=3, shadow=True,fancybox=True,prop={'size':18})
    n_years=1
    t_sampling=1 # sampling rate, 1 data per hour
    x = np.arange(1,8761*n_years, t_sampling) 
    y = np.hstack([np.array(datos.loc[idx['H2', flex, float(co2_lim)], :])]*n_years)
    n = len(x)
    y_fft=np.fft.fft(y)/n #n for normalization    
    frq=np.arange(0,1/t_sampling,1/(t_sampling*n))        
    period=np.array([1/f for f in frq])        
    ax2.semilogx(period[1:n//2],abs(y_fft[1:n//2])**2/np.max(abs(y_fft[1:n//2])**2), color=dic_color[co2_lim],
                 linewidth=2, label='CO$_2$ = '+dic_label[co2_lim])  
    ax2.legend(loc='center right', shadow=True,fancybox=True,prop={'size':18})
    #ax2.set_yticks([0, 0.1, 0.2])
    #ax2.set_yticklabels(['0', '0.1', '0.2'])
    plt.text(26, 0.95, 'day', horizontalalignment='left', color='dimgrey', fontsize=14)
    plt.text(24*7+20, 0.95, 'week', horizontalalignment='left', color='dimgrey', fontsize=14)
    plt.text(24*30+20, 0.95, 'month', horizontalalignment='left', color='dimgrey', fontsize=14)
    if i==2:
        ax2.set_xticks([1, 10, 100, 1000, 10000])
        ax2.set_xticklabels(['1', '10', '100', '1000', '10000'])
        ax2.set_xlabel('cycling period (hours)')
    else: 
        ax2.set_xticks([])
name = r'\Fourier_transform_H2_summed.jpg'
plt.show()
plt.savefig(path+name,dpi=300,format='jpg') #bbox_inches='tight' 

#%% Battery Fourier Power Series
pathdata = r'C:\Users\ander\OneDrive - Aarhus universitet\Maskiningenioer\Kandidat\3. semester\PreProject Master\Network files'

flex= 'elec_s_37'  
line_limit = 'lv1.0'
co2_limit = '0.1'
solar = 'solar+p3-'
cost_dist='1'

#line_limit='0.125' 
co2_limits=['0.5', '0.2', '0.1', '0.05',  '0']

flexs = ['elec_s_37'] 
techs=['battery']

network_name= (flex + '_' + line_limit + '__' +'Co2L'+ co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')

network = pypsa.Network(network_name)         

datos = pd.DataFrame(index=pd.MultiIndex.from_product([pd.Series(data=techs, name='tech',),
                                                       pd.Series(data=flexs, name='flex',),
                                                       pd.Series(data=co2_limits, name='co2_limits',)]), 
                      columns=pd.Series(data=np.arange(0,8760), name='hour',))
idx = pd.IndexSlice
df = pd.DataFrame()
generators=pd.DataFrame()

for co2_limit in co2_limits:
        network_name= (flex + '_' + line_limit + '__' +'Co2L'+ co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')  
        network = pypsa.Network(network_name)
        generators = np.array(network.stores_t.p[network.stores.index[network.stores.carrier == 'battery']].sum(axis=1))
        datos.loc[idx['battery', flex ,co2_limit], :] = generators

# Save dataframe to pickled pandas object and csv file
datos.to_pickle(pathdata+'\data_for_figures/battery_summed_timeseries.pickle') 
datos.to_csv(pathdata+'\data_for_figures/battery_summed_timeseries.csv', sep=',') 


## The plot
##### Figure of the Fourier transform for the PHS charging patterns
datos=pd.read_csv(pathdata+'\data_for_figures/battery_summed_timeseries.csv', sep=',', header=0, index_col=(0,1,2))


plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.figure(figsize=(10, 10))
gs1 = gridspec.GridSpec(10, 1)
gs1.update(wspace=0.05)

ax1 = plt.subplot(gs1[0:3,0])
ax1.set_ylabel('Battery charge level')
ax1.set_xlabel('hour')
ax1.set_xlim(0,8760)
ax1.set_ylim(0,1)

flex='elec_s_37'#'elec_central' #'elec_only'

co2_limits=['0.5', '0.2', '0']
storage_names=['battery'] #,'battery','H2']
dic_color={'battery':'darkgreen'}
storage_names=['battery'] #,'battery','H2']
dic_color={'0.5':'olive','0.2':'darkgreen','0':'red'}
dic_label={'0.5':'50%','0.2':'20%','0':'0%'}
dic_alpha={'0.5':1,'0.2':1,'0':1}
dic_linewidth={'0.5':2,'0.2':2,'0':2}

for i,co2_lim in enumerate(co2_limits):
    ax2 = plt.subplot(gs1[4+2*i:6+2*i,0])    
    ax2.set_xlim(1,10000)
    ax2.set_ylim(0,1.2)
    plt.axvline(x=24, color='lightgrey', linestyle='--')
    plt.axvline(x=24*7, color='lightgrey', linestyle='--')
    plt.axvline(x=24*30, color='lightgrey', linestyle='--')
    plt.axvline(x=8760, color='lightgrey', linestyle='--')   
    ax1.plot(np.arange(0,8760), datos.loc[idx['battery', flex, float(co2_lim)], :]/np.max(datos.loc[idx['battery', flex, float(co2_lim)], :]), 
             color=dic_color[co2_lim], alpha=dic_alpha[co2_lim], linewidth=dic_linewidth[co2_lim],
             label='CO$_2$='+dic_label[co2_lim])
    ax1.legend(loc=(0.2, 1.05), ncol=3, shadow=True,fancybox=True,prop={'size':18})
    n_years=1
    t_sampling=1 # sampling rate, 1 data per hour
    x = np.arange(1,8761*n_years, t_sampling) 
    y = np.hstack([np.array(datos.loc[idx['battery', flex, float(co2_lim)], :])]*n_years)
    n = len(x)
    y_fft=np.fft.fft(y)/n #n for normalization    
    frq=np.arange(0,1/t_sampling,1/(t_sampling*n))        
    period=np.array([1/f for f in frq])        
    ax2.semilogx(period[1:n//2],abs(y_fft[1:n//2])**2/np.max(abs(y_fft[1:n//2])**2), color=dic_color[co2_lim],
                 linewidth=2, label='CO$_2$ = '+dic_label[co2_lim])  
    ax2.legend(loc='center right', shadow=True,fancybox=True,prop={'size':18})
    #ax2.set_yticks([0, 0.1, 0.2])
    #ax2.set_yticklabels(['0', '0.1', '0.2'])
    plt.text(26, 0.95, 'day', horizontalalignment='left', color='dimgrey', fontsize=14)
    plt.text(24*7+20, 0.95, 'week', horizontalalignment='left', color='dimgrey', fontsize=14)
    plt.text(24*30+20, 0.95, 'month', horizontalalignment='left', color='dimgrey', fontsize=14)
    if i==2:
        ax2.set_xticks([1, 10, 100, 1000, 10000])
        ax2.set_xticklabels(['1', '10', '100', '1000', '10000'])
        ax2.set_xlabel('cycling period (hours)')
    else: 
        ax2.set_xticks([])
name = r'\Fourier_transform_battery_summed.jpg'
plt.show()
plt.savefig(path+name,dpi=300,format='jpg') #bbox_inches='tight' 

#%% Home Battery Fourier Power Series
pathdata = r'C:\Users\ander\OneDrive - Aarhus universitet\Maskiningenioer\Kandidat\3. semester\PreProject Master\Network files'

flex= 'elec_s_37'  
line_limit = 'lv1.0'
co2_limit = '0.1'
solar = 'solar+p3-'
cost_dist='1'

#line_limit='0.125' 
co2_limits=['0.5', '0.2', '0.1', '0.05',  '0']

flexs = ['elec_s_37'] 
techs=['home battery']

network_name= (flex + '_' + line_limit + '__' +'Co2L'+ co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')

network = pypsa.Network(network_name)         

datos = pd.DataFrame(index=pd.MultiIndex.from_product([pd.Series(data=techs, name='tech',),
                                                       pd.Series(data=flexs, name='flex',),
                                                       pd.Series(data=co2_limits, name='co2_limits',)]), 
                      columns=pd.Series(data=np.arange(0,8760), name='hour',))
idx = pd.IndexSlice
df = pd.DataFrame()
generators=pd.DataFrame()

for co2_limit in co2_limits:
        network_name= (flex + '_' + line_limit + '__' +'Co2L'+ co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')  
        network = pypsa.Network(network_name)
        generators = np.array(network.stores_t.p[network.stores.index[network.stores.carrier == 'home battery']].sum(axis=1))
        datos.loc[idx['home battery', flex ,co2_limit], :] = generators

# Save dataframe to pickled pandas object and csv file
datos.to_pickle(pathdata+'\data_for_figures/homebattery_summed_timeseries.pickle') 
datos.to_csv(pathdata+'\data_for_figures/homebattery_summed_timeseries.csv', sep=',') 


## The plot
##### Figure of the Fourier transform for the PHS charging patterns
datos=pd.read_csv(pathdata+'\data_for_figures/homebattery_summed_timeseries.csv', sep=',', header=0, index_col=(0,1,2))


plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.figure(figsize=(10, 10))
gs1 = gridspec.GridSpec(10, 1)
gs1.update(wspace=0.05)

ax1 = plt.subplot(gs1[0:3,0])
ax1.set_ylabel('Home battery charge level')
ax1.set_xlabel('hour')
ax1.set_xlim(0,8760)
ax1.set_ylim(0,1)

flex='elec_s_37'#'elec_central' #'elec_only'

co2_limits=['0.5', '0.2', '0']
storage_names=['home battery'] #,'battery','H2']
dic_color={'home battery':'darkgreen'}
storage_names=['home battery'] #,'battery','H2']
dic_color={'0.5':'olive','0.2':'darkgreen','0':'red'}
dic_label={'0.5':'50%','0.2':'20%','0':'0%'}
dic_alpha={'0.5':1,'0.2':1,'0':1}
dic_linewidth={'0.5':2,'0.2':2,'0':2}

for i,co2_lim in enumerate(co2_limits):
    ax2 = plt.subplot(gs1[4+2*i:6+2*i,0])    
    ax2.set_xlim(1,10000)
    ax2.set_ylim(0,1.2)
    plt.axvline(x=24, color='lightgrey', linestyle='--')
    plt.axvline(x=24*7, color='lightgrey', linestyle='--')
    plt.axvline(x=24*30, color='lightgrey', linestyle='--')
    plt.axvline(x=8760, color='lightgrey', linestyle='--')   
    ax1.plot(np.arange(0,8760), datos.loc[idx['home battery', flex, float(co2_lim)], :]/np.max(datos.loc[idx['home battery', flex, float(co2_lim)], :]), 
             color=dic_color[co2_lim], alpha=dic_alpha[co2_lim], linewidth=dic_linewidth[co2_lim],
             label='CO$_2$='+dic_label[co2_lim])
    ax1.legend(loc=(0.2, 1.05), ncol=3, shadow=True,fancybox=True,prop={'size':18})
    n_years=1
    t_sampling=1 # sampling rate, 1 data per hour
    x = np.arange(1,8761*n_years, t_sampling) 
    y = np.hstack([np.array(datos.loc[idx['home battery', flex, float(co2_lim)], :])]*n_years)
    n = len(x)
    y_fft=np.fft.fft(y)/n #n for normalization    
    frq=np.arange(0,1/t_sampling,1/(t_sampling*n))        
    period=np.array([1/f for f in frq])        
    ax2.semilogx(period[1:n//2],abs(y_fft[1:n//2])**2/np.max(abs(y_fft[1:n//2])**2), color=dic_color[co2_lim],
                 linewidth=2, label='CO$_2$ = '+dic_label[co2_lim])  
    ax2.legend(loc='center right', shadow=True,fancybox=True,prop={'size':18})
    #ax2.set_yticks([0, 0.1, 0.2])
    #ax2.set_yticklabels(['0', '0.1', '0.2'])
    plt.text(26, 0.95, 'day', horizontalalignment='left', color='dimgrey', fontsize=14)
    plt.text(24*7+20, 0.95, 'week', horizontalalignment='left', color='dimgrey', fontsize=14)
    plt.text(24*30+20, 0.95, 'month', horizontalalignment='left', color='dimgrey', fontsize=14)
    if i==2:
        ax2.set_xticks([1, 10, 100, 1000, 10000])
        ax2.set_xticklabels(['1', '10', '100', '1000', '10000'])
        ax2.set_xlabel('cycling period (hours)')
    else: 
        ax2.set_xticks([])
name = r'\Fourier_transform_homebattery_summed.jpg'
plt.show()
plt.savefig(path+name,dpi=300,format='jpg') #bbox_inches='tight' 
