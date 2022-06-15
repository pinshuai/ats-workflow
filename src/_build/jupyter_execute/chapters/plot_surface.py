#!/usr/bin/env python
# coding: utf-8

# <a href="https://githubtocolab.com/pinshuai/modvis/blob/master/examples/notebooks/plot_surface.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>

# # Plot surface variables

# In[ ]:


# skip this if package has already been installed
get_ipython().system('pip install modvis')


# In[2]:


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


visfile = xdmf.VisFile(model_dir, domain='surface', load_mesh=True)


# # plot surface ponded depth

# In[5]:


fig, ax, tpc = pv.plot_surface_data(visfile, var_name="surface-ponded_depth", 
                                    log = True, vmin=0.01, vmax=4, 
                                    time_slice=0)


# # plot ET

# In[6]:


fig, ax, tpc = pv.plot_surface_data(visfile, var_name="surface-total_evapotranspiration", 
                                    log = False, vmin=0, vmax=3, 
                                    time_slice= "2015-10-01")


# # plot snow cover

# In[7]:


fig, ax, tpc = pv.plot_surface_data(visfile, var_name="surface-area_fractions.cell.1", 
                                    vmin=0, vmax=1, time_slice=0)

