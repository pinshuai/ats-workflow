#!/usr/bin/env python
# coding: utf-8

# # Write ATS input file

# We now generate three input files -- two for spinup (steadystate solution and cyclic steadystate solution) and one for transient runs.
# 
# * Input files: ATS xml files
#   - `{WATERSHED_NAME}_spinup-steadystate.xml` the steady-state solution based on uniform application of mean rainfall rate
#   - `{WATERSHED_NAME}_spinup-cyclic_steadystate.xml` the cyclic steady state based on typical years
#   - `{WATERSHED_NAME}_transient.xml` the forward model

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150


# In[2]:


import os, yaml, pickle, datetime
import pandas as pd

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

# ats_input_spec library, to be moved to amanzi_xml
import ats_input_spec
import ats_input_spec.public
import ats_input_spec.io

# amanzi_xml, included in AMANZI_SRC_DIR/tools/amanzi_xml
import amanzi_xml.utils.io as aio
import amanzi_xml.utils.search as asearch
import amanzi_xml.utils.errors as aerrors


# In[3]:


import modvis
from modvis import ats_xml


# In[4]:


name = 'CoalCreek' # name the domain, used in filenames, etc
config_fname = '../../data/examples/CoalCreek/processed/config.yaml'


# ## load configuration

# In[5]:


# Load the dictionary from the file
with open(config_fname, 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


# In[6]:


config['spinup_steadystate_rundir'] = os.path.join('1-spinup_steadystate')
config['spinup_cyclic_rundir'] = os.path.join('2-spinup_cyclic')
config['transient_rundir'] = os.path.join('3-transient')

config['spinup_steadystate_template'] = os.path.join('..', '..', 'model', 'inputs', 'spinup_steadystate-template.xml')
config['spinup_cyclic_template'] = os.path.join('..', '..', 'model', 'inputs', 'spinup_cyclic-template.xml')
config['transient_template'] = os.path.join('..', '..', 'model', 'inputs', 'transient-template.xml')

config['watershed_specific_xml'] = os.path.join('..', '..', 'model', 'inputs', f'{name}_specific.xml')
config['spinup_steadystate_xml'] = os.path.join('..', '..', 'model', 'inputs', f'{name}_spinup_steadystate.xml')
config['spinup_cyclic_xml'] = os.path.join('..', '..', 'model', 'inputs', f'{name}_spinup_cyclic.xml')
config['transient_xml'] = os.path.join('..', '..', 'model', 'inputs', f'{name}_transient.xml')

config['latitude [deg]'] = 39 # latitude of watershed in degree, used to determine incident radiation


# In[7]:


config


# In[8]:


# nlcd_indices = config['nlcd_indices']
nlcd_labels = config['nlcd_labels']
subcatchment_labels = config['catchment_labels']
ls = config['labeled_sets']
ss = config['side_sets']
mean_precip = config['mean_precip [m s^-1]']
start_date = config['start_date']
end_date = config['end_date']


# In[9]:


# load subsurface properties
subsurface_props = pd.read_csv(config['subsurface_properties_filename'], index_col='ats_id')


# ## Generate watershed-specific properties

# Create a dummy xml file with watershed specific content (e.g., meshes, domain, forcing, and land covers) that will replace sections within the template files.

# In[10]:


# create the main list, this file is used for filling the template file
main_list = ats_xml.get_main(config, subsurface_props, nlcd_labels,
                             labeled_sets = ls, side_sets = ss,
                             subcatchment_labels=subcatchment_labels,
                            )
main_xml = ats_input_spec.io.to_xml(main_list)

# save generated xml 
ats_input_spec.io.write(main_list, config['watershed_specific_xml'])


# ## Write input files
# 
# Replace template files with generated watershed specific properties. This also sets the start and end date of the simulations, and creates directories for each run.

# - `{name}_spinup_steadystate.xml`: For the first file, we load a spinup template and write the needed quantities into that file, saving it to the appropriate run directory.  Note there is no DayMet or land cover or LAI properties needed for this run.  The only property that is needed is the domain-averaged, mean annual rainfall rate.  We then take off some for ET (note too wet spins up faster than too dry, so don't take off too much...).
# 
# - `{name}_spinup_cyclic.xml`: For the second file, we load a transient run template.  This file needs the basics, plus DayMet and LAI as the "typical year data".  Also we set the run directory that will be used for the steadystate run.
# 
# - `{name}_transient.xml`: For the third file, we load a transient run template as well.  This file needs the basics, DayMet with the actual data, and we choose for this run to use the MODIS typical year.  MODIS is only available for 2002 on, so if we didn't need 1980-2002 we could use the real data, but for this run we want a longer record.

# In[11]:


# create a steady-state run
ats_xml.write_spinup_steadystate(config, main_xml, mean_precip = mean_precip)

# make sure the cyclic ends near Oct. 1
ats_xml.write_transient(config, main_xml,             
                start_date = start_date, end_date=end_date,cyclic_steadystate=True, 
               )

# create the fully-heterogeneous runs
ats_xml.write_transient(config, main_xml,                
                 start_date = start_date, end_date=end_date,cyclic_steadystate=False, 
               )


# In[12]:


with open(config_fname, 'w') as f:
    yaml.dump(config, f)


# In[ ]:




