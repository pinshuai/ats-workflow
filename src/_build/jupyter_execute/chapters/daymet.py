#!/usr/bin/env python
# coding: utf-8

# # Download Daymet

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# Daymet provides gridded meteorological data for North American at 1km spatial resolution with daily timestep from 1980 ~ present. [website](https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1328) and [user guide](https://daac.ornl.gov/DAYMET/guides/Daymet_V3_CFMosaics.html)
# 
# Available variables:
# 
# | Variable | Description (units) |
# | ---- | ---- |
# | tmax | Daily maximum 2-meter air temperature (°C) |
# | tmin | Daily minimum 2-meter air temperature (°C) |
# | prcp | Daily total precipitation (mm/day) |
# | srad | Incident shortwave radiation flux density (W/m2) |
# | vp   | Water vapor pressure (Pa) |
# | swe  | Snow water equivalent (kg/m2) |
# | dayl | Duration of the daylight period (seconds) |
# 
# Notes:
#  - The Daymet calendar is based on a standard calendar year. All Daymet years, including leap years, have 1 - 365 days. For leap years, the Daymet database includes leap day (February 29) and values for December 31 are discarded from leap years to maintain a 365-day year.
#  
#  - DayMet's incident shortwave radiation is the "daylit" radiation.  To get the daily average radiation, one must multiply by daylit fraction, given by dayl / 86400.

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150


# In[3]:


import logging
import numpy as np
import rasterio
import fiona
import os

import watershed_workflow
import watershed_workflow.ui
import watershed_workflow.daymet

watershed_workflow.ui.setup_logging(1,None)


# In[4]:


watershed_shapefile = 'Coweeta/input_data/coweeta_basin.shp'


# ## import watershed

# In[5]:


crs, watershed = watershed_workflow.get_split_form_shapes(watershed_shapefile)
logging.info(f'crs: {crs}')

bounds = watershed.exterior()
bounds


# ## Download

# returned raw data has `dim(nband, ncol, nrow)`

# In[6]:


startdate = "1-1980"
enddate = "365-1980"


# In[7]:


# setting vars = None to download all available variables
raw, x, y = watershed_workflow.daymet.collectDaymet(bounds, crs=crs, 
                                                    start=startdate, end=enddate)


# ## Reproject Daymet CRS
# 
# Reproject daymet CRS to the same as the watershed. This is necessary if watershed meshes are using watershed CRS.

# In[8]:


new_x, new_y, new_extent, new_dat, daymet_profile = \
        watershed_workflow.daymet.reproj_Daymet(x, y, raw, dst_crs=crs)


# ## plot Daymet

# In[9]:


ivar = 'tmax'
islice = 100
fig, axes = plt.subplots(1, 2)

ax = axes[0]
extent = rasterio.transform.array_bounds(daymet_profile['height'], daymet_profile['width'], daymet_profile['transform']) # (x0, y0, x1, y1)
plot_extent = extent[0], extent[2], extent[1], extent[3]

iraster = raw[ivar][islice, :, :]

with fiona.open(watershed_shapefile, mode='r') as fid:
    bnd_profile = fid.profile
    bnd = [r for (i,r) in fid.items()]
daymet_crs = watershed_workflow.crs.daymet_crs()

# convert to destination crs
native_crs = watershed_workflow.crs.from_fiona(bnd_profile['crs'])
reproj_bnd = watershed_workflow.warp.shape(bnd[0], native_crs, daymet_crs)
reproj_bnd_shply = watershed_workflow.utils.shply(reproj_bnd)

cax = ax.matshow(iraster, extent=plot_extent, alpha=1)
ax.plot(*reproj_bnd_shply.exterior.xy, 'r')
ax.set_title("Raw Daymet")


ax = axes[1]
extent = new_extent # (x0, y0, x1, y1)
plot_extent = extent[0], extent[2], extent[1], extent[3] # (x0, x1, y0, y1)

iraster = new_dat[ivar][islice, :, :]

# set nodata to NaN to avoid plotting
iraster[iraster == -9999] = np.nan

watershed_workflow.plot.hucs(watershed, crs, ax=ax, color='r', linewidth=1)
im = ax.matshow(iraster, extent=plot_extent)
ax.set_title("Reprojected Daymet")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.25, 0.02, 0.5])
fig.colorbar(im, cax=cbar_ax)


# # save daymet

# ## Write to HDF5
# 
# Write raw daymet data to a single HDF5 file.

# In[10]:


watershed_workflow.daymet.writeHDF5(new_dat, new_x, new_y, 
                         watershed_workflow.daymet.getAttrs(bounds.bounds, startdate, enddate), 
                         os.path.join('Coweeta','output_data', 'watershed_daymet-raw.h5'))


# ## Write to ATS format
# 
# This will write daymet in a format that ATS can read. E.g., this will partition precipitation into rain and snow, convert vapor pressure to relative humidity, get mean air temperature and so on.
# 
# - dout has dims of `(ntime, nrow, ncol)` or `(ntime, ny, nx)`

# In[11]:


dat_ats = watershed_workflow.daymet.writeATS(new_dat, new_x, new_y, 
                         watershed_workflow.daymet.getAttrs(bounds.bounds, startdate, enddate), 
                         os.path.join('Coweeta', 'output_data', 'watershed_daymet-ats.h5'))


# In[ ]:




