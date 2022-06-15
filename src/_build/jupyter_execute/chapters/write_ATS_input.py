#!/usr/bin/env python
# coding: utf-8

# # Write ATS input file

# We now generate three input files -- two for spinup (steadystate solution and cyclic steadystate solution) and one for transient runs.
# 
# Steadystate has its own physics, but cyclic steadystate and transient share a common set of physics.  Each have their own met data strategy.
# 
# The first step is to generate the sections of xml that will replace parts of the template files.  This is done prior to loading any templates to make clear that these are totally generated from scratch using the ats_input_spec tool.
# 
# Note that throughout, we will assume an additional level of folder nesting, e.g. runs will be completed in '../spinup-CoalCreek/run0', meaning that we have to append an extra '../' to the start of all filenames.  This makes it easier to deal with mistakes, continued runs, etc.

# In[33]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150


# In[34]:


import os, yaml, pickle
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


# In[36]:


name = 'NeversinkHeadwaters_11Catchments_cluster' # name the domain, used in filenames, etc
canopy = True
data_dir = "../data/"
input_dir = "../model/inputs/"
# mesh_dir = "../../data/meshes"

template_dir =input_dir + f"template_xml/master/ecohydro/"
out_dir = "../data/WW_output/"
pickle_dir = out_dir + "pickle/"
processed_dir = data_dir + "processed/"

# case = 'cyber'
# if canopy:
#     template_filename = template_dir + f"reactive-canopy-{case}-template.xml"
# else:
#     template_filename = template_dir + f"reactive-{case}-template.xml"


# In[38]:


with open(out_dir + f"{name}_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

config


# In[39]:


# a dictionary of outputs -- will include all filenames generated
# config = {}

# name = "NeversinkHeadwaters"
# input xmls
mesh_filename = os.path.join('../..', config['mesh_filename'])

# config['subsurface_properties_filename'] = data_dir + "WW_output/subsurface_prop.csv"
# input data
config['modis_typical_filename'] = processed_dir + f'NeversinkHeadwaters_MODIS_LAI_typical_1980_2020.h5'
config['daymet_filename'] = processed_dir + f'NeversinkHeadwaters_Daymet_1980_2020-UTM.h5'
config['daymet_spinup_filename'] = processed_dir + f'NeversinkHeadwaters_DayMet_typical_1980_2020-UTM.h5'
# run dirs
config[f'spinup_steadystate_{name}_rundir'] = f'../model/{name}_spinup_steadystate'
config[f'spinup_cyclic_{name}_rundir'] = f'../model/{name}_spinup_cyclic'
config[f'transient_{name}_rundir'] = f'../model/{name}_transient'

config[f'spinup_steadystate_{name}_template'] = template_dir + f'spinup_steadystate-template.xml'
config[f'spinup_cyclic_{name}_template'] = template_dir + f'spinup_cyclic-template.xml'
config[f'transient_{name}_template'] = template_dir + f'transient-template.xml'

config['generated_ats'] = input_dir + f'{name}_generated_ats.xml'
config[f'spinup_steadystate_{name}_filename'] = input_dir + f'{name}_spinup_steadystate.xml'
config[f'spinup_cyclic_{name}_filename'] = input_dir + f'{name}_spinup_cyclic.xml'
config[f'transient_{name}_filename'] = input_dir + f'{name}_transient.xml'
# if canopy:
#     config[f'spinup_cyclic_{name}_filename'] = input_dir + f'{name}_spinup_cyclic-reactive-canopy-{case}.xml'
#     config[f'transient_{name}_filename'] = input_dir + f'{name}_transient-reactive-canopy-{case}.xml'    
# else:
#     config[f'spinup_cyclic_{name}_filename'] = input_dir + f'{name}_spinup_cyclic-reactive-{case}.xml'
#     config[f'transient_{name}_filename'] = input_dir + f'{name}_transient-reactive-{case}.xml'


# subcatchment_labels = ['poly_bottom', 'poly_eastbranch', 'poly_westbranch']

config['latitude'] = 41 # in deg
config['mean_precip'] = 2.500629987318984e-08 # in m/s
latitude = config['latitude']
mean_precip = config['mean_precip']

start_day = 0
end_day = 3650 # in days


# In[41]:


with open(pickle_dir + f"{name}_subcatchment_labels.p", "rb") as f:
    subcatchment_labels = pickle.load(f)


# In[42]:


with open(pickle_dir + f"{name}_sidesets.p", "rb") as f:
    ls, ss = pickle.load(f)


# In[43]:


with open(pickle_dir + f"{name}_nlcd.p", "rb") as f:
    nlcd_labels_dict = pickle.load(f)


# In[44]:


if canopy:
    nlcd_indices, nlcd_labels = nlcd_labels_dict.keys(), nlcd_labels_dict.values()    


# In[16]:


# calculate the basin-averaged, annual-averaged precip rate
# precip_total = ats_typ['precipitation rain [m s^-1]'] + ats_typ['precipitation snow [m SWE s^-1]']
# mean_precip = precip_total.mean()
# mean_precip = 2.500629987318984e-08
# logging.info(f'Mean annual precip rate [m s^-1] = {mean_precip}')


# In[45]:


subsurface_props = pd.read_csv(config['subsurface_properties_filename'], index_col='ats_id')
# subsurface_props.loc[999, 'native_index'] = 999
# subsurface_props['native_index'] = subsurface_props['native_index'].astype(float).astype(int).astype(str)


# In[46]:


# add the subsurface and surface domains
#
# Note this also adds a "computational domain" region to the region list, and a vis spec 
# for "domain"
def add_domains(main_list, mesh_filename, surface_region='surface', snow=True, canopy=True):
    ats_input_spec.public.add_domain(main_list, 
                                 domain_name='domain', 
                                 dimension=3, 
                                 mesh_type='read mesh file',
                                 mesh_args={'file':mesh_filename})
    if surface_region:
        main_list['mesh']['domain']['build columns from set'] = surface_region    
    
        # Note this also adds a "surface domain" region to the region list and a vis spec for 
        # "surface"
        ats_input_spec.public.add_domain(main_list,
                                domain_name='surface',
                                dimension=2,
                                mesh_type='surface',
                                mesh_args={'surface sideset name':'surface'})
    if snow:
        # Add the snow and canopy domains, which are aliases to the surface
        ats_input_spec.public.add_domain(main_list,
                                domain_name='snow',
                                dimension=2,
                                mesh_type='aliased',
                                mesh_args={'target':'surface'})
    if canopy:
        ats_input_spec.public.add_domain(main_list,
                                domain_name='canopy',
                                dimension=2,
                                mesh_type='aliased',
                                mesh_args={'target':'surface'})


# The mafic potential seems wrong below!

# In[47]:


def add_land_cover(main_list, nlcd_indices, nlcd_labels):
    # next write a land-cover section for each NLCD type
    for index, nlcd_name in zip(nlcd_indices, nlcd_labels):
        # this will load default values instead of pulling from the template
        ats_input_spec.public.set_land_cover_default_constants(main_list, nlcd_name)

    land_cover_list = main_list['state']['initial conditions']['land cover types']
    # update some defaults
    # ['Other', 'Deciduous Forest', 'Evergreen Forest', 'Shrub/Scrub']
    # note, these are from the CLM Technical Note v4.5
    #
    # Rooting depth curves from CLM TN 4.5 table 8.3
    #
    # Note, the mafic potential values are likely pretty bad for the types of van Genuchten 
    # curves we are using (ETC -- add paper citation about this topic).  Likely they need
    # to be modified.  Note that these values are in [mm] from CLM TN 4.5 table 8.1, so the 
    # factor of 10 converts to [Pa]
    #
    # Note, albedo of canopy taken from CLM TN 4.5 table 3.1
    
    land_cover_list['Other']['rooting profile alpha [-]'] = 7.0
    land_cover_list['Other']['rooting profile beta [-]'] = 2.0
    land_cover_list['Other']['rooting depth max [m]'] = 0.5
    land_cover_list['Other']['mafic potential at fully closed stomata [Pa]'] = 2750000
    land_cover_list['Other']['mafic potential at fully open stomata [Pa]'] = 7400
    land_cover_list['Other']['Priestley-Taylor alpha of snow [-]'] = 1.2
    land_cover_list['Other']['Priestley-Taylor alpha of bare ground [-]'] = 1.0
    land_cover_list['Other']['Priestley-Taylor alpha of canopy [-]'] = 1.2
    land_cover_list['Other']['Priestley-Taylor alpha of transpiration [-]'] = 1.2
    
    land_cover_list['Evergreen Forest']['rooting profile alpha [-]'] = 7.0
    land_cover_list['Evergreen Forest']['rooting profile beta [-]'] = 2.0
    land_cover_list['Evergreen Forest']['rooting depth max [m]'] = 2.0
    land_cover_list['Evergreen Forest']['mafic potential at fully closed stomata [Pa]'] = 2500785
    land_cover_list['Evergreen Forest']['mafic potential at fully open stomata [Pa]'] = 647262
    land_cover_list['Evergreen Forest']['Priestley-Taylor alpha of snow [-]'] = 1.2
    land_cover_list['Evergreen Forest']['Priestley-Taylor alpha of bare ground [-]'] = 1.0
    land_cover_list['Evergreen Forest']['Priestley-Taylor alpha of canopy [-]'] = 1.2
    land_cover_list['Evergreen Forest']['Priestley-Taylor alpha of transpiration [-]'] = 1.2

    land_cover_list['Deciduous Forest']['rooting profile alpha [-]'] = 6.0
    land_cover_list['Deciduous Forest']['rooting profile beta [-]'] = 2.0
    land_cover_list['Deciduous Forest']['rooting depth max [m]'] = 2.0
    land_cover_list['Deciduous Forest']['mafic potential at fully closed stomata [Pa]'] = 2196768
    land_cover_list['Deciduous Forest']['mafic potential at fully open stomata [Pa]'] = 343245
    land_cover_list['Deciduous Forest']['Priestley-Taylor alpha of snow [-]'] = 1.2
    land_cover_list['Deciduous Forest']['Priestley-Taylor alpha of bare ground [-]'] = 1.0
    land_cover_list['Deciduous Forest']['Priestley-Taylor alpha of canopy [-]'] = 1.2
    land_cover_list['Deciduous Forest']['Priestley-Taylor alpha of transpiration [-]'] = 1.2
    
    land_cover_list['Shrub/Scrub']['rooting profile alpha [-]'] = 7.0
    land_cover_list['Shrub/Scrub']['rooting profile beta [-]'] = 1.5
    land_cover_list['Shrub/Scrub']['rooting depth max [m]'] = 0.5
    land_cover_list['Shrub/Scrub']['mafic potential at fully closed stomata [Pa]'] = 4197396
    land_cover_list['Shrub/Scrub']['mafic potential at fully open stomata [Pa]'] = 813981
    land_cover_list['Shrub/Scrub']['Priestley-Taylor alpha of snow [-]'] = 1.2
    land_cover_list['Shrub/Scrub']['Priestley-Taylor alpha of bare ground [-]'] = 1.0
    land_cover_list['Shrub/Scrub']['Priestley-Taylor alpha of canopy [-]'] = 1.2
    land_cover_list['Shrub/Scrub']['Priestley-Taylor alpha of transpiration [-]'] = 1.2


# In[48]:


# add soil sets: note we need a way to name the set, so we use, e.g. SSURGO-MUKEY.
def soil_set_name(ats_id):
    if ats_id == 999:
        return 'bedrock'
    source = subsurface_props.loc[ats_id]['source']
    native_id = subsurface_props.loc[ats_id]['native_index']
    if type(native_id) in [tuple,list]:
        native_id = native_id[0]
    elif type(native_id) is str:
        native_id = native_id.replace('(', '').replace(')', '').split(',')[0]
    else:
        raise("native_id is not a known type!")
    return f"{source}-{native_id}"


# In[49]:


# get an ATS "main" input spec list -- note, this is a dummy and is not used to write any files yet
def get_main(mesh_filename, subsurface_props, subcatchment_labels=None, snow=True, canopy=True):
    main_list = ats_input_spec.public.get_main()
    
    # get PKs
    flow_pk = ats_input_spec.public.add_leaf_pk(main_list, 'flow', main_list['cycle driver']['PK tree'], 
                                            'richards-spec')

    # add the mesh and all domains
    # mesh_filename = os.path.join('..', config['mesh_filename'])
    add_domains(main_list, mesh_filename, canopy=canopy)
    
    if canopy:
        # add labeled sets
        try:
            for i in ls:
                ats_input_spec.public.add_region_labeled_set(main_list, i.name, i.setid, mesh_filename, i.entity)
            for j in ss:
                ats_input_spec.public.add_region_labeled_set(main_list, j.name, j.setid, mesh_filename, 'FACE')
        except:
            logging.info("no sidesets provided. adding surface and bottom only")
            for iname,id in zip(['surface','bottom'], [2,1]):
                ats_input_spec.public.add_region_labeled_set(main_list, iname, id, mesh_filename, 'FACE') 
           
        # add land cover
        add_land_cover(main_list, nlcd_indices, nlcd_labels)
    else:
        for iname,id in zip(['surface','bottom'], [2,1]):
            ats_input_spec.public.add_region_labeled_set(main_list, iname, id, mesh_filename, 'FACE')

    # add soil material ID regions, porosity, permeability, and WRMs
    for ats_id in subsurface_props.index:
        props = subsurface_props.loc[ats_id]
        set_name = soil_set_name(ats_id)
        
        if props['van Genuchten n [-]'] < 1.5:
            smoothing_interval = 0.01
        else:
            smoothing_interval = 0.0
        
        ats_input_spec.public.add_soil_type(main_list, set_name, ats_id, mesh_filename,
                                            float(props['porosity [-]']),
                                            float(props['permeability [m^2]']), 
                                            1.e-9, # pore compressibility, maybe too large?
                                            float(props['van Genuchten alpha [Pa^-1]']),
                                            float(props['van Genuchten n [-]']),
                                            float(props['residual saturation [-]']),
                                            float(smoothing_interval))
        
    # add observations for each subcatchment for transient runs
    # this will add default observed variables instead of getting those from template
    
    obs = ats_input_spec.public.add_observations_water_balance(main_list, region="computational domain", 
                                                               surface_region= "surface domain")
    
    # add a few additional observations
    # make sure to use "external sides" as region for calculating net gw flux!
    
    ats_input_spec.public.add_observeable(obs, 'SWE [m]', 'snow-water_equivalent', 
                                          'surface domain', 'average','cell', time_integrated=False)
    ats_input_spec.public.add_observeable(obs, 'max ponded depth [m]', 'surface-ponded_depth', 
                                          'surface domain', 'maximum','cell', time_integrated=False)
    ats_input_spec.public.add_observeable(obs, 'water to surface [m d^-1]', 'canopy-throughfall_drainage_rain', 
                                          'surface domain', 'average','cell', time_integrated=True)
    ats_input_spec.public.add_observeable(obs, 'snow to surface [m SWE d^-1]', 'canopy-throughfall_drainage_snow', 
                                          'surface domain', 'average','cell', time_integrated=True)
    ats_input_spec.public.add_observeable(obs, 'canopy drainage [m d^-1]', 'canopy-drainage', 
                                          'surface domain', 'average','cell', time_integrated=True)
    ats_input_spec.public.add_observeable(obs, 'total evapotranspiration [m d^-1]', 'surface-total_evapotranspiration', 
                                          'surface domain', 'average','cell', time_integrated=True)    
    
    if subcatchment_labels is not None:
        for region in subcatchment_labels:
            obs = ats_input_spec.public.add_observations_water_balance(main_list, region, 
                                                                 outlet_region = region + ' outlet')
            # add a few additional observations
            ats_input_spec.public.add_observeable(obs, 'SWE [m]', 'snow-water_equivalent', 
                                                  region + ' surface', 'average','cell', time_integrated=False)
            ats_input_spec.public.add_observeable(obs, 'max ponded depth [m]', 'surface-ponded_depth', 
                                                  region + ' surface', 'maximum','cell', time_integrated=False)
            ats_input_spec.public.add_observeable(obs, 'water to surface [m d^-1]', 'canopy-throughfall_drainage_rain', 
                                                  region + ' surface', 'average','cell', time_integrated=True)
            ats_input_spec.public.add_observeable(obs, 'snow to surface [m SWE d^-1]', 'canopy-throughfall_drainage_snow', 
                                                  region + ' surface', 'average','cell', time_integrated=True)
            ats_input_spec.public.add_observeable(obs, 'canopy drainage [m d^-1]', 'canopy-drainage', 
                                                  region + ' surface', 'average','cell', time_integrated=True)
            ats_input_spec.public.add_observeable(obs, 'total evapotranspiration [m d^-1]', 'surface-total_evapotranspiration', 
                                                  region + ' surface', 'average','cell', time_integrated=True)            
        
    
    
    return main_list


# In[50]:


def populate_basic_properties(xml, main_xml, canopy=True, homogeneous_wrm=False, homogeneous_poro=False, homogeneous_perm=False):
    """This function updates an xml object with the above properties for mesh, regions, soil props, and lc props"""
    # find and replace the mesh list
    mesh_i = next(i for (i,el) in enumerate(xml) if el.get('name') == 'mesh')
    xml[mesh_i] = asearch.child_by_name(main_xml, 'mesh')

    # find and replace the regions list
    region_i = next(i for (i,el) in enumerate(xml) if el.get('name') == 'regions')
    xml[region_i] = asearch.child_by_name(main_xml, 'regions')

    # find and replace the WRMs list -- note here we only replace the inner "WRM parameters" because the
    # demo has this in the PK, not in the field evaluators list
    if not homogeneous_wrm:
        wrm_list = asearch.find_path(xml, ['PKs', 'water retention evaluator'])
        wrm_i = next(i for (i,el) in enumerate(wrm_list) if el.get('name') == 'WRM parameters')
        wrm_list[wrm_i] = asearch.find_path(main_xml, ['PKs','water retention evaluator','WRM parameters'])

    fe_list = asearch.find_path(xml, ['state', 'field evaluators'])

    # find and replace porosity, permeability
    if not homogeneous_poro:
        poro_i = next(i for (i,el) in enumerate(fe_list) if el.get('name') == 'base_porosity')
        fe_list[poro_i] = asearch.find_path(main_xml, ['state', 'field evaluators', 'base_porosity'])

    if not homogeneous_perm:
        perm_i = next(i for (i,el) in enumerate(fe_list) if el.get('name') == 'permeability')
        fe_list[perm_i] = asearch.find_path(main_xml, ['state', 'field evaluators', 'permeability'])

    if canopy:
        # find and replace land cover
        consts_list = asearch.find_path(xml, ['state', 'initial conditions'])
        try:
            lc_i = next(i for (i,el) in enumerate(consts_list) if el.get('name') == 'land cover types')
        except StopIteration:
            pass
        else:
            consts_list[lc_i] = asearch.find_path(main_xml, ['state', 'initial conditions', 'land cover types'])


# In[51]:


def create_unique_name(name, homogeneous_wrm=False, homogeneous_poro=False, homogeneous_perm=False):
    suffix = '_h'
    if homogeneous_perm:
        suffix += 'K'
    if homogeneous_poro:
        suffix += 'p'
    if homogeneous_wrm:
        suffix += 'w'
    if suffix == '_h':
        suffix = ''
    return name + suffix
        


# In[52]:


def write_spinup_steadystate(name, template_file, **kwargs):
    name = create_unique_name(name, **kwargs)
    filename = config[f'spinup_steadystate_{name}_filename']
    logging.info(f'Writing spinup steadystate: {filename}')
    
    # write the spinup xml file
    # load the template file
    # xml = aio.fromFile(template_dir + 'spinup_steadystate-template.xml')
    xml = aio.fromFile(template_file)
    # populate basic properties for mesh, regions, and soil properties
    populate_basic_properties(xml, main_xml, **kwargs)

    # set the mean avg source as 60% of mean precip
    precip_el = asearch.find_path(xml, ['state', 'field evaluators', 'surface-precipitation', 
                                        'function-constant', 'value'])
    precip_el.setValue(mean_precip * .6)

    # # update mismatch outlet region
    # par = asearch.find_path(xml, ['regions', 'poly_bottom surface outlet', 'label'])
    # par.setValue('10012')
    # par = asearch.find_path(xml, ['regions', 'surface domain outlet', 'label'])
    # par.setValue('10012')
    # par = asearch.find_path(xml, ['regions', 'poly_westbranch surface outlet', 'label'])
    # par.setValue('10010')     
    
    # write to disk
    # config[f'spinup_steadystate_{name}_filename'] = model_dir + f'input_xml/spinup_steadystate.xml'
    aio.toFile(xml, config[f'spinup_steadystate_{name}_filename'])

    # make a run directory
    # config[f'spinup_steadystate_{name}_rundir'] = f'../model/{name}_spinup_steadystate'
    # try:
    #     os.mkdir(config[f'spinup_steadystate_{name}_rundir'])
    # except FileExistsError:
    #     pass


# In[53]:


def write_transient(name, template_filename, canopy=True, cyclic_steadystate=False, start_year=1980, 
                    end_year=2020, start_day=None, end_day=None, **kwargs):
    # make a unique name based on options
    name = create_unique_name(name, **kwargs)
    

    if cyclic_steadystate:
        prefix = 'spinup_cyclic'
        start_year = 1980
        end_year = 1990
        previous = 'spinup_steadystate'
        runnum = 'run1'     
    else:
        prefix = 'transient'
        previous = 'spinup_cyclic'
        runnum = 'run2'
        
    filename = config[f'{prefix}_{name}_filename']
    logging.info(f'Writing {prefix}: {filename}')
    # template_filename = template_dir + f'{prefix}-template.xml'
    
    # load the template file
    xml = aio.fromFile(template_filename)

    # populate basic properties for mesh, regions, and soil properties
    populate_basic_properties(xml, main_xml, canopy=False, **kwargs)

    # update the DayMet filenames
    # wind speed uses default?
    if cyclic_steadystate:
        daymet_filename = config['daymet_spinup_filename']
    else:
        daymet_filename = config['daymet_filename']
        
    for var in ['surface-incoming_shortwave_radiation',
                'surface-precipitation_rain',
                'snow-precipitation',
                'surface-air_temperature',
                'surface-relative_humidity',
                'surface-temperature',
                'canopy-temperature']:
        try:
            par = asearch.find_path(xml, ['state', 'field evaluators', var, 'file'])
        except aerrors.MissingXMLError:
            pass
        else:
            par.setValue(os.path.join('../..', daymet_filename))
    
    if canopy:
    # update the LAI filenames
        for par in asearch.findall_path(xml, ['canopy-leaf_area_index', 'file']):
            par.setValue(os.path.join('../..', config['modis_typical_filename']))
    
    # update the start and end time -- start at Oct 1 of year 0, end 10 years later
    if start_day is None:
        start_day = 274 + 365*(start_year - 1980)
    par = asearch.find_path(xml, ['cycle driver', 'start time'])
    par.setValue(start_day)

    if end_day is None:
        end_day = 274 + 365*(end_year - 1980)
    par = asearch.find_path(xml, ['cycle driver', 'end time'])
    par.setValue(end_day)
    
    # update the restart filenames
    for var in asearch.findall_path(xml, ['initial condition', 'restart file']):
        var.setValue(os.path.join('..', config[f'{previous}_{name}_rundir'], 'checkpoint_final.h5'))

    # update the observations list
    obs = next(i for (i,el) in enumerate(xml) if el.get('name') == 'observations')
    xml[obs] = asearch.child_by_name(main_xml, 'observations')
   
    # update surface-incident-shortwave-radiation
    par = asearch.find_path(xml, ['state', 'field evaluators', 'surface-incident_shortwave_radiation', 'latitude [degrees]'])
    par.setValue(latitude)   
    
    # write to disk and make a directory for running the run
    filename = config[f'{prefix}_{name}_filename']
    # rundir = config[f'{prefix}_{name}_rundir']

    aio.toFile(xml, filename)
    # try:
    #     os.mkdir(rundir)
    # except FileExistsError:
    #     pass



# For the first file, we load a spinup template and write the needed quantities into that file, saving it to the appropriate run directory.  Note there is no DayMet or land cover or LAI properties needed for this run.  The only property that is needed is the domain-averaged, mean annual rainfall rate.  We then take off some for ET (note too wet spins up faster than too dry, so don't take off too much...).

# Create a dummy xml file with the content that template file can use.

# In[54]:


# create the main list, this file is used for filling the template file
main_list = get_main(mesh_filename, subsurface_props, subcatchment_labels=subcatchment_labels, canopy=canopy)
ats_input_spec.io.write(main_list, config['generated_ats'])
main_xml = ats_input_spec.io.to_xml(main_list)


# For the second file, we load a transient run template.  This file needs the basics, plus DayMet and LAI as the "typical year data".  Also we set the run directory that will be used for the steadystate run.

# For the third file, we load a transient run template as well.  This file needs the basics, DayMet with the actual data, and we choose for this run to use the MODIS typical year.  MODIS is only available for 2002 on, so if we didn't need 1980-2002 we could use the real data, but for this run we want a longer record.

# In[55]:


# create the fully-heterogeneous runs
# if include_heterogeneous:
write_spinup_steadystate(name, config[f'spinup_steadystate_{name}_template'])
# make sure the cyclic ends near Oct. 1
write_transient(name,  config[f'spinup_cyclic_{name}_template'], canopy=canopy, 
                cyclic_steadystate=True, 
                # start_day=start_day, end_day=end_day,
                end_year=1990
               )
write_transient(name, config[f'transient_{name}_template'], canopy=canopy, 
                cyclic_steadystate=False, start_day=None, end_day=None)

# # create homogeneous runs
# if include_homogeneous:
#     write_spinup_steadystate(name, homogeneous_wrm=True, homogeneous_poro=True, homogeneous_perm=True)
#     write_transient(name, True, homogeneous_wrm=True, homogeneous_poro=True, homogeneous_perm=True)
#     write_transient(name, False, homogeneous_wrm=True, homogeneous_poro=True, homogeneous_perm=True)
    
# if include_homogeneous_wrm:
#     write_spinup_steadystate(name, homogeneous_wrm=True, homogeneous_poro=False, homogeneous_perm=False)
#     write_transient(name, True, homogeneous_wrm=True, homogeneous_poro=False, homogeneous_perm=False)
#     write_transient(name, False, homogeneous_wrm=True, homogeneous_poro=False, homogeneous_perm=False)
    
# if include_homogeneous_wrm_porosity:
#     write_spinup_steadystate(name, homogeneous_wrm=True, homogeneous_poro=True, homogeneous_perm=False)
#     write_transient(name, True, homogeneous_wrm=True, homogeneous_poro=True, homogeneous_perm=False)
#     write_transient(name, False, homogeneous_wrm=True, homogeneous_poro=True, homogeneous_perm=False)
    
# if include_homogeneous_wrm_permeability:
#     write_spinup_steadystate(name, homogeneous_wrm=True, homogeneous_poro=False, homogeneous_perm=True)
#     write_transient(name, True, homogeneous_wrm=True, homogeneous_poro=False, homogeneous_perm=True)
#     write_transient(name, False, homogeneous_wrm=True, homogeneous_poro=False, homogeneous_perm=True)


# In[56]:


with open(out_dir + f"{name}_config.yaml", 'w') as f:
    yaml.dump(config, f)


# In[57]:


logging.info('this workflow is a total success')


# ## Next Steps
# 
# Changes needed to update the xml file:
# 
# - change `mass_flux` to `water_flux` to work with the master branch
# - change `net groundwater flux` applied boundary from `"computational domain boundary"` to `"external_sides"` 

# In[ ]:




