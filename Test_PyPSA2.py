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
line_limits='lv1.0' #['lv1.0','lv1.1','lv1.2','lv1.5','lv2.0']
cost_dists='0.1' #['0.1','0.5','1','2','10']
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

# for cost_dist in cost_dists:
#     for line_limit in line_limits:
#         for co2_limit in co2_limits:    
#                     network_name= (flex+ '_' + line_limit + '__' +'Co2L'+co2_limit+ '-' + solar +'dist'+cost_dist+'_'+'2030'+'.nc')
#                     if cost_dist=='0.1' and line_limit=='lv1.0':
#                     d1["n" + str(cost_dist) +str(line_limit) +str(co2_limit)] = pypsa.Network(network_name) 
#                     d2["generators"+ str(cost_dist) +str(line_limit) +str(co2_limit)] = d1["n" + str(cost_dist) +str(line_limit) +str(co2_limit)].generators.groupby("carrier")["p_nom_opt"].sum()
#                     d3["sizes" + str(cost_dist) +str(line_limit) +str(co2_limit)] = [d2["generators"+ str(cost_dist) +str(line_limit) +str(co2_limit)]['gas'].sum(),
#                                                                              d2["generators"+ str(cost_dist) +str(line_limit) +str(co2_limit)]['offwind-ac'].sum(),
#                                                                              d2["generators"+ str(cost_dist) +str(line_limit) +str(co2_limit)]['offwind-dc'].sum(),
#                                                                              d2["generators"+ str(cost_dist) +str(line_limit) +str(co2_limit)]['onwind'].sum(),
#                                                                              d2["generators"+ str(cost_dist) +str(line_limit) +str(co2_limit)]['ror'].sum(),
#                                                                              d2["generators"+ str(cost_dist) +str(line_limit) +str(co2_limit)]['solar'].sum(),
#                                                                              d2["generators"+ str(cost_dist) +str(line_limit) +str(co2_limit)]['solar rooftop'].sum()]

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

#%% 
## Fourier Power series for all of Europe ##




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






