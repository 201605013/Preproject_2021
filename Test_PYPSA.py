#!/usr/bin/env python
# coding: utf-8

# # Aarhus University - Fall 2020 - Renewable Energy Systems (RES) project 
# 
# This notebook includes the steps to optimize the capacity and dispatch of generators in the power system of one country.
# Make sure that you understand every step in this notebook. For the project of the course Renewable Energy Systems (RES) you need to deliver a report including the sections described at the end of this notebook.

# In[1]:


import pypsa
#pandas package is very useful to work with imported data, time series, matrices ...
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# We start by creating the network. In this example, the country is modelled as a single node, so the network will only include one bus.
# 
# We select the year 2015 and set the hours in that year as snapshots.
# 
# We select a country, in this case, Spain (ESP), and add one node (electricity bus) to the network.

n = pypsa.Network("elec_s_37_lv1.0__Co2L0.05-solar+p3-dist10_2030.nc")

# In[2]:

x  = 10    

capacity = n.generators

generators = n.generators.groupby("carrier")["p_nom_opt"].sum()

print(generators)

storage = n.storage_units.groupby("carrier")["p_nom_opt"].sum()

print(storage)

labels = ['gas',
          'offwind-ac',
          'offwind-dc',
          'onwind',
          'ror',
          'solar', 
          'solar rooftop']
sizes = [generators['gas'].sum(),
         generators['offwind-ac'].sum(),
         generators['offwind-dc'].sum(),
         generators['onwind'].sum(),
         generators['ror'].sum(),
         generators['solar'].sum(),
         generators['solar rooftop'].sum()]

colors=['brown','blue','dodgerblue','green','yellow','orange','red']

plt.pie(sizes, 
        colors=colors, 
        labels=labels, 
        wedgeprops={'linewidth':0})
plt.axis('equal')

plt.title('Electricity mix', y=1.07)

# Loading the network in a proper way


flex= 'elec_s_37'  
lv = 'lv1.0'
co2_limit = 'Co2L0.1'
solar = 'solar+p3-dist'

    
network_name = (flex + '_' + lv + '__' + co2_limit+ '-' + solar +'1'+'_'+'2030'+'.nc')

network = pypsa.Network(network_name)         

co2_limits=['0.5', '0.2', '0.1', '0.05',  '0'] #, '0.025']

for co2_limit in co2_limits:      
        network_name= (flex+ '_' + lv + '__' +'Co2L'+co2_limit+ '-' + solar +'1'+'_'+'2030'+'.nc')        
        print(network_name)
        n = pypsa.Network(network_name) 
        generators = n.generators.groupby("carrier")["p_nom_opt"].sum()
        print(generators)
        
        plt.figure()
        labels = ['gas',
                  'offwind-ac',
                  'offwind-dc',
                  'onwind',
                  'ror',
                  'solar', 
                  'solar rooftop']
        sizes = [generators['gas'].sum(),
                 generators['offwind-ac'].sum(),
                 generators['offwind-dc'].sum(),
                 generators['onwind'].sum(),
                 generators['ror'].sum(),
                 generators['solar'].sum(),
                 generators['solar rooftop'].sum()]


        colors=['brown','blue','dodgerblue','green','yellow','orange','red']

        plt.pie(sizes, 
                colors=colors, 
                labels=labels, 
                wedgeprops={'linewidth':0})
        plt.axis('equal')

        plt.title('Electricity mix', y=1.07)
        plt.show()
        
    


# In[3]

n.plot()

for c in n.iterate_components(list(n.components.keys())[2:]):
    print("Component '{}' has {} entries".format(c.name,len(c.df)))


# In[4] Temporal resolution

print(len(n.snapshots))

# In[5] Static component data

print(n.lines.head())

print(n.generators.head())

print(n.storage_units.head())

# In[6] Time-varying component data (input data)

n.loads_t.p_set.sum(axis=1).plot(figsize=(15,3))

n.generators_t.p_max_pu.head()

#plotting the solar generation for Italy in july
n.generators_t.p_max_pu.loc["2013-07","IT0 0 solar"].plot(figsize=(15,3))

#plotting the onshore wind generation for Estonia in july
n.generators_t.p_max_pu.loc["2013-07","ES0 0 onwind"].plot(figsize=(15,3))


# In[7] Total Annual System Cost

# Total system cost, in billion euroes

n.objective / 1e9

# In[8] Transmission line Expansion

(n.lines.s_nom_opt - n.lines.s_nom).head(5)


# In[9] Optimal Generator/Storage Capacities

# Optimal generator units
n.generators.groupby("carrier").p_nom_opt.sum() /1e3

# Optimal storage units
n.storage_units.groupby("carrier").p_nom_opt.sum() /1e3


# In[10] Energy storage over time

(n.storage_units_t.state_of_charge.sum(axis=1).resample('D').mean() / 1e6).plot(figsize=(15,3))


(n.storage_units_t.state_of_charge["2013-07"].filter(like="PHS",axis=1)).plot(figsize=(15,3))

# In[11] plotting energy networks on maps

import cartopy.crs as ccrs

loading = (n.lines_t.p0.abs().mean().sort_index() / (n.lines.s_nom_opt*n.lines.s_max_pu).sort_index()).fillna(0.)

# PlateCarree, Mercator, Orthographic

fig,ax = plt.subplots(
                subplot_kw = {"projection": ccrs.PlateCarree()}
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




# In[4]

print(n.generators.p_nom_opt)
print(n.storage_units.p_nom_opt)
print(n.links.p_nom_opt)
print(n.generators_t.p)

network = pypsa.Network()
hours_in_2015 = pd.date_range('2015-01-01T00:00Z','2015-12-31T23:00Z', freq='H')
network.set_snapshots(hours_in_2015)

network.add("Bus","electricity bus")


# The load is represented by the historical electricity demand in 2015 with hourly resolution. 
# 
# The file with historical hourly electricity demand for every European country is available in the data folder.
# 
# The electricity demand time series were obtained from ENTSOE through the very convenient compilation carried out by the Open Power System Data (OPSD). https://data.open-power-system-data.org/time_series/

# In[3]:


# load electricity demand data
df_elec = pd.read_csv('data/electricity_demand.csv', sep=';', index_col=0) # in MWh
print(df_elec['DEU'].head())


# In[4]:


# add load to the bus
network.add("Load",
            "load", 
            bus="electricity bus", 
            p_set=df_elec['DEU'])


# In the optimization, we will minimize the annualized system costs.
# 
# We will need to annualize the cost of every generator, we build a function to do it.

# In[5]:


def annuity(n,r):
    """Calculate the annuity factor for an asset with lifetime n years and
    discount rate of r, e.g. annuity(20,0.05)*20 = 1.6"""

    if r > 0:
        return r/(1. - 1./(1.+r)**n)
    else:
        return 1/n


# We include solar PV and onshore wind generators. 
# 
# The capacity factors representing the availability of those generators for every European country can be downloaded from the following repositories (select 'optimal' for PV and onshore for wind). 
# 
# https://zenodo.org/record/3253876#.XSiVOEdS8l0
# 
# https://zenodo.org/record/2613651#.XSiVOkdS8l0
# 
# We include also Open Cycle Gas Turbine (OCGT) generators
# 
# The cost assumed for the generators are the same as in the paper https://doi.org/10.1016/j.enconman.2019.111977 (open version:  https://arxiv.org/pdf/1906.06936.pdf)

# In[6]:


# add the different carriers, only gas emits CO2
network.add("Carrier", "gas", co2_emissions=0.19) # in t_CO2/MWh_th
network.add("Carrier", "onshorewind")
network.add("Carrier", "solar")
network.add("Carrier", "offshorewind")


# add onshore wind generator
df_onshorewind = pd.read_csv('data/onshore_wind_1979-2017.csv', sep=';', index_col=0)
CF_wind = df_onshorewind['DEU'][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]
capital_cost_onshorewind = annuity(20,0.07)*910000*(1+0.033) # in €/MW
network.add("Generator",
            "onshorewind",
            bus="electricity bus",
            p_nom_extendable=True,
            carrier="onshorewind",
            #p_nom_max=1000, # maximum capacity can be limited due to environmental constraints
            capital_cost = capital_cost_onshorewind,
            marginal_cost = 0,
            p_max_pu = CF_wind)

# add offshore wind generator
df_offshorewind = pd.read_csv('data/offshore_wind_1979-2017.csv', sep=';', index_col=0)
CF_offwind = df_offshorewind['DEU'][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]
capital_cost_offshorewind = annuity(30,0.07)*810000*(1+0.033) # in €/MW
network.add("Generator",
            "offshorewind",
            bus="electricity bus",
            p_nom_extendable=True,
            carrier="offshorewind",
            #p_nom_max=1000, # maximum capacity can be limited due to environmental constraints
            capital_cost = capital_cost_offshorewind,
            marginal_cost = 0,
            p_max_pu = CF_offwind)

# add solar PV generator
df_solar = pd.read_csv('data/pv_optimal.csv', sep=';', index_col=0)
CF_solar = df_solar['DEU'][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in network.snapshots]]
capital_cost_solar = annuity(25,0.07)*425000*(1+0.03) # in €/MW
network.add("Generator",
            "solar",
            bus="electricity bus",
            p_nom_extendable=True,
            carrier="solar",
            #p_nom_max=1000, # maximum capacity can be limited due to environmental constraints
            capital_cost = capital_cost_solar,
            marginal_cost = 0,
            p_max_pu = CF_solar)

# add OCGT (Open Cycle Gas Turbine) generator
capital_cost_OCGT = annuity(25,0.07)*560000*(1+0.033) # in €/MW
fuel_cost = 21.6 # in €/MWh_th
efficiency = 0.39
marginal_cost_OCGT = fuel_cost/efficiency # in €/MWh_el
network.add("Generator",
            "OCGT",
            bus="electricity bus",
            p_nom_extendable=True,
            carrier="gas",
            #p_nom_max=1000,
            capital_cost = capital_cost_OCGT,
            marginal_cost = marginal_cost_OCGT)


# We solve the linear optimal power flow (lopf) using Gurobi as solver.
# 
# In this case, we are optimising the installed capacity and dispatch of every generator to minimize the total system cost.

# In[7]:


network.lopf(network.snapshots, 
             solver_name='gurobi')


# The result ('ok', 'optimal') indicates that the optimizer has found an optimal solution. 
# 
# The total cost can be read from the network objetive.

# In[21]:


print(network.objective/1000000) #in 10^6 €
print(network.objective/network.loads_t.p.sum()) # €/MWh


# The optimal capacity for every generator can be shown.

# In[23]:


network.generators.p_nom_opt # in MW


# We can plot now the dispatch of every generator during the first week of the year and the electricity demand.
# We import the matplotlib package which is very useful to plot results.
# 
# We can also plot the electricity mix.

# In[24]:

n1 = 100
n2 = 400

import matplotlib.pyplot as plt

plt.plot(network.loads_t.p['load'][n1:n2], color='black', label='demand')
plt.plot(network.generators_t.p['onshorewind'][n1:n2], color='blue', label='onshore wind')
plt.plot(network.generators_t.p['offshorewind'][n1:n2], color='yellow', label='offshore wind')
plt.plot(network.generators_t.p['solar'][n1:n2], color='orange', label='solar')
plt.plot(network.generators_t.p['OCGT'][n1:n2], color='brown', label='gas (OCGT)')
plt.legend(fancybox=True, shadow=True, loc='best')


# In[25]:


labels = ['onshore wind',
          'offshore wind',
          'solar', 
          'gas (OCGT)']
sizes = [network.generators_t.p['onshorewind'].sum(),
         network.generators_t.p['offshorewind'].sum(),
         network.generators_t.p['solar'].sum(),
         network.generators_t.p['OCGT'].sum()]

colors=['blue','yellow', 'orange', 'brown']

plt.pie(sizes, 
        colors=colors, 
        labels=labels, 
        wedgeprops={'linewidth':0})
plt.axis('equal')

plt.title('Electricity mix', y=1.07)


# We can add a global CO2 constraint and solve again.

# In[26]:


co2_limit=400000 #tonCO2
network.add("GlobalConstraint",
            "co2_limit",
            type="primary_energy",
            carrier_attribute="co2_emissions",
            sense="<=",
            constant=co2_limit)

network.lopf(network.snapshots, 
             solver_name='gurobi')


# In[27]:


network.generators.p_nom_opt #in MW


# In[28]:


import matplotlib.pyplot as plt

n1 = 1000
n2 = 1100

plt.plot(network.loads_t.p['load'][n1:n2], color='black', label='demand')
plt.plot(network.generators_t.p['onshorewind'][n1:n2], color='blue', label='onshore wind')
plt.plot(network.generators_t.p['offshorewind'][n1:n2], color='yellow', label='offshore wind')
plt.plot(network.generators_t.p['solar'][n1:n2], color='orange', label='solar')
plt.plot(network.generators_t.p['OCGT'][n1:n2], color='brown', label='gas (OCGT)')
plt.legend(fancybox=True, shadow=True, loc='best')


# In[29]:


labels = ['onshore wind',
          'offshore wind',
          'solar', 
          'gas (OCGT)']
sizes = [network.generators_t.p['onshorewind'].sum(),
         network.generators_t.p['offshorewind'].sum(),
         network.generators_t.p['solar'].sum(),
         network.generators_t.p['OCGT'].sum()]

colors=['blue','yellow', 'orange', 'brown']


plt.pie(sizes, 
        colors=colors, 
        labels=labels, 
        wedgeprops={'linewidth':0})
plt.axis('equal')

plt.title('Electricity mix', y=1.07)


# ## PROJECT INSTRUCTIONS
# 
# Based on the previous example, you are asked to carry out the following tasks:
# 
# A. Choose a different country/region and calculate the optimal capacities for renewable and non-renewable generators. You can add as many technologies as you want. Remember to provide a reference for the cost assumptions. Plot the dispatch time series for a week in summer and winter. Plot the annual electricity mix. Use the duration curves or the capacity factor to investigate the contribution from different technologies. 
# 
# B. Investigate how sensitive is the optimum capacity mix to the global CO2 constraint. E.g., plot the generation mix as a function of the CO2 constraint that you impose. Search for the CO2 emissions in your country (today or in 1990) and refer the emissions allowance to that historical data. 
# 
# C. Investigate how sensitive are your results to the interannual variability of solar and wind generation. Plot the average capacity and variability obtained for every generator using different weather years. 
# 
# D. Add some storage technology/ies and investigate how they behave and what is their impact on the optimal system configuration. 
# 
# E. Discuss what strategies is your system using to balance the renewable generation at different time scales (intraday, seasonal, etc.) 
# 
# F. Select on target for decarbonizatio (i.e., one CO2 allowance limit). What is the CO2 price required to achieve that decarbonization level? Search for information on the existing CO2 tax in your country (if any) and discuss your result. 
# 
# G. Connect your country with, at least, two neighbour countries. You can assume that the capacities in the neighbours are fixed or cooptimize the whole system. You can also include fixed interconnection capacities or cooptimize them with the generators capacities. Discuss your results.
# 
# H. Connect the electricity sector with another sector such as heating or transport, and cooptimize the two sectors. Discuss your results.
# 
# I. Finally, select one topic that is under discussion in your region. Design and implement some experiment to obtain relevant information regarding that topic. E.g. 
# 
# [-] What are the consequences if Denmark decides not to install more onshore wind? 
# 
# [-] Would it be more expensive if France decides to close its nuclear power plants? 
# 
# [-] What will be the main impacts of the Viking link?
# 
# Write a short report (maximum 10 pages) including your main findings.

# 

# 
# 
# 
# _TIP 1: You can add a link with the following code_
# 
# The efficiency will be 1 if you are connecting two countries and different from one if, for example, you are connecting the electricity bus to the heating bus using a heat pump.
# Setting p_min_pu=-1 makes the link reversible.
# 

# In[30]:


network.add("Link",
             'country a - country b',
             bus0="electricity bus country a",
             bus1="electricity bus country b",
             p_nom_extendable=True, # capacity is optimised
             p_min_pu=-1,
             length=600, # length (in km) between country a and country b
             capital_cost=400*600) # capital cost * length 


# 
# _TIP 2: You can check the KKT multiplier associated with the constraint with the following code_
# 

# In[42]:


print(network.global_constraints.constant) #CO2 limit (constant in the constraint)

print(network.global_constraints.mu) #CO2 price (Lagrance multiplier in the constraint)


# In[ ]:




