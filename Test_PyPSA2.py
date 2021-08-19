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
path = r'C:\Users\ander\OneDrive - Aarhus universitet\Maskiningenioer\Kandidat\3. semester\PreProject Master\Network files\Figures testing'


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
fig, (ax1, ax2, ax3) = plt.subplots(1, 3,dpi = 300,sharey=True)
fig.suptitle('Installed capacity vs. CO2 constrain and Transmission expansion')
df1.plot.bar(ax=ax1,rot=25 )
ax1.set_xlabel('Carrier')
ax1.set_ylabel('Installed capacity [MW]')
ax1.set_title('Carrier capacity vs. CO2 emmisions - lv1.0')
ax1.set_ylim(0,700e3)
ax1.yaxis.grid()
ax1.legend(['CO2 50%','CO2 20%','CO2 10%','CO2 5%', 'CO2 0%'])
#plt.rc('grid', linestyle="--", color='gray')
#ax1.legend(frameon = True, ncol = 5, shadow=True, bbox_to_anchor=(0.5, 1.25), loc='upper center', title = '% CO2 emmesion compared to 1990')

df2.plot.bar(ax=ax2,rot=25 )
ax2.set_xlabel('Carrier')
ax2.set_ylabel('Installed capacity [MW]')
ax2.set_title('Carrier capacity vs. CO2 emmisions - lv1.1')
ax2.set_ylim(0,700e3)
ax2.legend(['CO2 50%','CO2 20%','CO2 10%','CO2 5%', 'CO2 0%'])
ax2.yaxis.grid()

df3.plot.bar(ax=ax3,rot=25 )
ax3.set_xlabel('Carrier')
ax3.set_ylabel('Installed capacity [MW]')
ax3.set_title('Carrier capacity vs. CO2 emmisions - lv2.0')
ax3.set_ylim(0,700e3)
ax3.legend(['CO2 50%','CO2 20%','CO2 10%','CO2 5%', 'CO2 0%'])
ax3.yaxis.grid()

# Save the figure in the selected path
name = r'\testing.jpg'
plt.show()
plt.savefig(path+name, dpi=300,format='jpg' ,bbox_inches='tight')   

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
cost_dist=['0.1','0.5','1','2','10']

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

dftotcost=pd.Series()

for cost_dist in cost_dists:
    dftotcost[cost_dist]=dftot.filter(like=str(cost_dist)+'lv')
### ARBEJDER HER!!!!


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






