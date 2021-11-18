# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 09:42:04 2021

@author: Anders
"""

import pypsa
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("bmh")


n = pypsa.Network("elec_s_37_lv2.0__Co2L0-solar+p3-dist0.1_2030.nc")

n = pypsa.Network('elec_s_5_ec_lcopt_Co2L-24H.nc')

n = pypsa.Network('elec_s_9_ec_lcopt_Co2L-168H0.nc')

n = pypsa.Network('elec_s_37.nc')


n.plot()

# Generators
print(n.generators.p_nom_opt)
generators=n.generators.p_nom_opt

# Storage
print(n.storage_units.p_nom_opt)

# Links
print(n.links.p_nom_opt)

# Generator output time series
print(n.generators_t.p)



print(n.global_constraints.constant) #CO2 limit (constant in the constraint)

print(n.global_constraints.mu) #CO2 price (Lagrance multiplier in the constraint)

# Plot of the network
n.plot()

for c in n.iterate_components(list(n.components.keys())[2:]):
    print("Component '{}' has {} entries".format(c.name,len(c.df)))


n.loads_t.p_set.sum(axis=1).plot(figsize=(15,3))

# Total Annual System Costs (billion euros p.a.)
n.objective/1e9

# Transmission Line Expansion
(n.lines.s_nom_opt-n.lines.s_nom).head(5)

# Optimal generator storage capacities # GW
n.generators.groupby("carrier").p_nom_opt.sum()/1e3
n.storage_units.groupby("carrier").p_nom_opt.sum()/1e3


# Energy Storage
(n.storage_units_t.state_of_charge.sum(axis=1).resample('D').mean() / 1e6).plot(figsize=(15,3))


n.plot()

pypsa.plot.plot(n)



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


# TESTING
n.generators.p_nom_opt["AT0 0 solar"]
n.generators_t.p["AT0 0 solar"]["2013-01-01 00:00":"2013-01-01 23:00"].plot.area(figsize = (9,4))


##---------------- Plots testing ----------------##
# Distributed vs centralized renewable generation in the European power system
# Analysis of optimal rooftop solar vs ground-mounted solar plants for different cost assumptions
# for the distribution grids, transmission, and CO2 emissions constraints.

# For different cost assumptions














