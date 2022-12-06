import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt


import netCDF4 as nc
import numpy as np
import pandas as pd
import os

##TEMPORARY###############
os.chdir(r"C:\Users\THom\Documents\GitHub\CS229_Project")
##############

dataPath = 'Weather Data/VIIRS-Land_v001-preliminary_NPP13C1_S-NPP_20180101_c20220418024242.nc'
data = nc.Dataset(dataPath)


west, south, east, north = (
    -156,
    18.979,
    -154.9,
    20.3
             )

df = pd.DataFrame(
    {'Latitude': [19.4069, 19.4721],
     'Longitude': [-155.2834, -155.5922]}
)

ax = df.plot.scatter(
    "Longitude", "Latitude", s=0.5, color='r'
)

ghent_img, ghent_ext = cx.bounds2img(west,
                                     south,
                                     east,
                                     north,
                                     ll=True,
                                     source=cx.providers.Stamen.TerrainBackground
                                    )

cx.add_basemap(
    ax,
    source = ghent_img
)











f, ax = plt.subplots(1)
ax.imshow(ghent_img, extent=ghent_ext)




# gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.Longitude, df.Latitude))
# gdf.plot(color='red')
plt.show()

print('e')