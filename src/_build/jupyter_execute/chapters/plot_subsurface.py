#!/usr/bin/env python
# coding: utf-8

# <a href="https://githubtocolab.com/pinshuai/modvis/blob/master/examples/notebooks/plot_subsurface.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>

# # Plot subsurface variables

# In[1]:


# skip this if package has already been installed
get_ipython().system('pip install modvis')


# In[2]:


import numpy as np
import modvis.ats_xdmf as xdmf
import modvis.plot_vis_file as pv

model_dir = "../data/coalcreek"


# Download the sample data when running on `Google Colab`

# In[3]:


import os
if not os.path.exists(model_dir):
  get_ipython().system('git clone https://github.com/pinshuai/modvis.git')
  get_ipython().run_line_magic('cd', './modvis/examples/notebooks')


# # import vis data

# In[4]:


visfile = xdmf.VisFile(model_dir, domain=None, load_mesh=True, columnar=True)


# # plot subsurface satuation

# ## single column

# In[5]:


fig, ax = pv.plot_column_data(visfile, var_name = "saturation_liquid", cmap = "coolwarm", 
                              col_ind=0)


# ## single layer

# In[6]:


fig, ax = pv.plot_layer_data(visfile, var_name = "saturation_liquid", 
                             layer_ind = 0, time_slice= "2015-10-01",
                          cmap = "coolwarm")


# # plot groundwater table

# ## single column

# In[7]:


ihead = pv.plot_column_head(visfile, col_ind = 0)


# ## GW surface

# In[8]:


fig, ax, tpc = pv.plot_gw_surface(visfile, time_slice = 0, contour = True, 
                                  contourline = True, nlevel = np.arange(2700,3665, 50), 
                                  colorbar = True,
                                 )


# # Volumetric water content

# In[9]:


fig, ax, tpc = pv.plot_water_content(visfile, layer_ind = 0, 
                                     vmin = 0.1, vmax = 0.22, cmap = 'turbo',
                                     time_slice=0)


# In[ ]:




