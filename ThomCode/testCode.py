import contextily as cx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

##TEMPORARY###############
os.chdir(r"C:\Users\THom\Documents\GitHub\CS229_Project")
##############

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

# ax = df.plot.scatter(
#     "Longitude", "Latitude", s=0.5, color='r'
# )

ghent_img, ghent_ext = cx.bounds2img(west,
                                     south,
                                     east,
                                     north,
                                     ll=True,
                                     source=cx.providers.Stamen.TerrainBackground
                                    )

# cx.add_basemap(
#     ax,
#     source = ghent_img
# )


LatScaleF = 1.7372*10**7/156.061628
LongScaleF = 2.144*10**6 / 18.9100782

f, ax = plt.subplots(1)
ax.imshow(ghent_img, extent=ghent_ext)
# ax.scatter([-155.4001667*LatScaleF],[19.466662*LongScaleF])




# gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.Longitude, df.Latitude))
# gdf.plot(color='red')
plt.show()

print('e')