#[ ........................................................... ]#
#[
#[ Check particle balance in a Gkeyll multiblock gyrokinetic simulation.
#[
#[ This script assumes the existence of the following files:
#[   - <sim_name>-<species_name>_fdot_integrated_moms.gkyl
#[   - <sim_name>-<species_name>_integrated_moms.gkyl
#[   - <sim_name>-<species_name>_source_integrated_moms.gkyl if using a source.
#[   - <sim_name>-<species_name>_bflux_<direction><side>_integrated_HamiltonianMoments.gkyl
#[       if non-periodic, non-zero-flux boundaries are used.
#[   - <sim_name>-dt.gkyl
#[ for each block.
#[
#[ Manaure Francisquez.
#[
#[ ........................................................... ]#

import numpy as np
import postgkyl as pg
import matplotlib.pyplot as plt
import os
import glob

#data_dir        = './gk_multib_step_sol_2x2v_p1/cfl0p5/' #[ Where Gkeyll data is located.
#simulation_name = 'gk_multib_step_sol_2x2v_p1' #[ Name of Gkeyll simulation.
data_dir        = './cons_data/hl2a' #[ Where Gkeyll data is located.
simulation_name = 'split_hl2a_coresource' #[ Name of Gkeyll simulation.
#simulation_name = 'gk_multib_asdex_2x2v_p1' #[ Name of Gkeyll simulation.
species_names    = ['elc', 'ion'] #[ Name of the particle species of interest.

plot_balance        = True #[ Balance of between various terms.
plot_relative_error = True #[ Conservation relative error.

save_fig_to_file = False #[ Output a figure file?.
fig_file_dir     = './' #[ Where to place figure written out.
fig_file_format  = '.png' #[ Can be .png, .pdf, .ps, .eps, .svg.


#[ ............... End of user inputs (MAYBE) ..................... ]#

#[ Some RGB colors. These are MATLAB-like.
defaultBlue    = [0, 0.4470, 0.7410]
defaultOrange  = [0.8500, 0.3250, 0.0980]
defaultGreen   = [0.4660, 0.6740, 0.1880]
defaultPurple  = [0.4940, 0.1840, 0.5560]
defaultRed     = [0.6350, 0.0780, 0.1840]
defaultSkyBlue = [0.3010, 0.7450, 0.9330]
grey           = [0.5, 0.5, 0.5]
#[ Colors in a single array.
defaultColors = [defaultBlue,defaultOrange,defaultGreen,defaultPurple,defaultRed,defaultSkyBlue,grey,'black']

#[ LineStyles in a single array.
lineStyles = ['-','--',':','-.','None','None','None','None']
markers    = ['None','None','None','None','o','d','s','+']

#[ Some fontsizes used in plots.
xyLabelFontSize       = 17
titleFontSize         = 17
colorBarLabelFontSize = 17
tickFontSize          = 14
legendFontSize        = 14

#[ Set the font size of the ticks to a given size.
def setTickFontSize(axIn,fontSizeIn):
  axIn.tick_params(axis='both',labelsize=fontSizeIn)
  offset_txt = axIn.yaxis.get_offset_text() # Get the text object
  offset_txt.set_size(fontSizeIn) # # Set the size.
  offset_txt = axIn.xaxis.get_offset_text() # Get the text object
  offset_txt.set_size(fontSizeIn) # # Set the size.

#[ Check if a file exists............. ]#
def does_file_exist(fileIn):
  if os.path.exists(fileIn):
     return True
  else:
     return False
#[ .................................... ]#

#[ Read data and time stamps from a DynVector.............. ]#
def read_dyn_vector(dataFile):
  pgData = pg.GData(dataFile)  #[ Read data with pgkyl.
  time   = pgData.get_grid()  #[ Time stamps of the simulation.
  val    = pgData.get_values()  #[ Data values.
  return np.squeeze(time), np.squeeze(val)
#[ ......................................................... ]#

#[ Labels used to identify boundary flux files.
edges = ["lower","upper"]
dirs = ["x","y","z"]

#[ ............... End common utilities ..................... ]#

#[ Read the Hamiltonian moment of df/dt, the source and the particle fluxes.

if plot_balance:
  #[ Read the Hamiltonian moment of df/dt, the source and the particle fluxes.

  #[ Number of blocks.
  fdot_file_list = glob.glob(data_dir + '/' + simulation_name + '_b*-' + species_names[0] + '_fdot_integrated_moms.gkyl')
  num_blocks = len(fdot_file_list)

  for bI in range(num_blocks):
    #[ Read field energy data.
    data_path = data_dir + '/' + simulation_name + '_b%d-'%bI
    field_file = data_path + 'field_energy_dot.gkyl'
    time_field_dot, field_dot_pb = read_dyn_vector(field_file)
    #[ Add over blocks.
    if bI == 0:
      field_dot = field_dot_pb
    else:
      field_dot += field_dot_pb

    for sI, species_name in enumerate(species_names):
      data_path = data_dir + '/' + simulation_name + '_b%d-' + species_name
      file = data_path % bI 
  
      #[ Read data.
      time_fdot, fdot_pb = read_dyn_vector(file + '_fdot_integrated_moms.gkyl')

      source_file = file + '_source_integrated_moms.gkyl'
      has_source = does_file_exist(source_file)
      if has_source:
        time_src, src_pb = read_dyn_vector(source_file)

      nbflux = 0
      time_bflux, bflux_pb = list(), list()
      has_bflux = False
      for d in dirs:
        for e in edges:
          bflux_file = file + '_bflux_' + d + e + '_integrated_HamiltonianMoments.gkyl'
          has_bflux_at_boundary = does_file_exist(bflux_file)
          if has_bflux_at_boundary:
            time_bflux_tmp, bflux_pb_tmp = read_dyn_vector(bflux_file)
            time_bflux.append(time_bflux_tmp)
            bflux_pb.append(bflux_pb_tmp)
            has_bflux = has_bflux or has_bflux_at_boundary
            nbflux += 1

      #[ Select the Hamiltonian moment.
      fdot_pb = fdot_pb[:,2]
      if has_source:
        src_pb = src_pb[:,2]
        src_pb[0] = 0.0 #[ Set source=0 at t=0 since we don't have fdot and bflux then.
      else:
        src_pb = 0.0*fdot_pb

      if has_bflux:
        for i in range(nbflux):
          bflux_pb[i] = bflux_pb[i][:,2]
      
        time_bflux_tot = time_bflux[0]
        bflux_pb_tot = bflux_pb[0] #[ Total boundary flux loss.
        for i in range(1,nbflux):
          bflux_pb_tot += bflux_pb[i]
      else:
        bflux_pb_tot = 0.0*fdot_pb

      #[ Add over blocks.
      if sI==0 and bI == 0:
        fdot = fdot_pb
        src = src_pb
        bflux_tot = bflux_pb_tot
      else:
        fdot += fdot_pb
        src += src_pb
        bflux_tot += bflux_pb_tot
  
  #[ Compute the error.
  mom_err = src - bflux_tot - (fdot - field_dot)
  
  #[ Plot each contribution.
  figProp1a = (7.5, 4.5)
  ax1aPos   = [0.09, 0.15, 0.87, 0.78]
  fig1a     = plt.figure(figsize=figProp1a)
  ax1a      = fig1a.add_axes(ax1aPos)
  
  hpl1a = list()
  hpl1a.append(ax1a.plot([-1.0,1.0], [0.0,0.0], color='grey', linestyle=':', linewidth=1))
  hpl1a.append(ax1a.plot(time_fdot, fdot, color=defaultColors[0], linestyle=lineStyles[0], linewidth=2))

  legendStrings = [r'$\dot{f}$']
  if has_source:
    hpl1a.append(ax1a.plot(time_src, src, color=defaultColors[2], linestyle=lineStyles[2], linewidth=2))
    legendStrings.append(r'$\mathcal{S}$')

  if has_bflux:
    hpl1a.append(ax1a.plot(time_bflux_tot, -bflux_tot, color=defaultColors[1], linestyle=lineStyles[1], linewidth=2))
    legendStrings.append(r'$-\int_{\partial \Omega}\mathrm{d}\mathbf{S}\cdot\mathbf{\dot{R}}f$')

  hpl1a.append(ax1a.plot(time_field_dot, field_dot, color=defaultColors[4], linestyle=':', linewidth=2, marker='+',markevery=8))
  legendStrings.append(r'$\dot{\phi}$')

  hpl1a.append(ax1a.plot(time_fdot, mom_err, color=defaultColors[3], linestyle=lineStyles[3], linewidth=2))
  legendStrings.append(r'$E_{\dot{\mathcal{N}}}=\mathcal{S}-\int_{\partial \Omega}\mathrm{d}\mathbf{S}\cdot\mathbf{\dot{R}}f-\dot{f}$')

  ax1a.set_xlabel(r'Time ($s$)',fontsize=xyLabelFontSize, labelpad=+4)
  ax1a.set_title(r'Particle balance',fontsize=titleFontSize)
  ax1a.set_xlim( time_fdot[0], time_fdot[-1] )
  ax1a.legend([hpl1a[i][0] for i in range(1,len(hpl1a))], legendStrings, fontsize=legendFontSize, frameon=False)
  setTickFontSize(ax1a,tickFontSize)
  
  if save_fig_to_file:
    plt.savefig(fig_file_dir+simulation_name+'_particle_balance'+fig_file_format)
  else:
    plt.show()

#[ .......................................................... ]#

if plot_relative_error:
  #[ Plot the error normalized for different time steps.
  
  time_dt, dt = read_dyn_vector(data_dir + '/' + simulation_name + '-dt.gkyl')
  
  #[ Number of blocks.
  fdot_file_list = glob.glob(data_dir + '/' + simulation_name + '_b*-' + species_names[0] + '_fdot_integrated_moms.gkyl')
  num_blocks = len(fdot_file_list)

  for bI in range(num_blocks):
    #[ Read field energy data.
    data_path = data_dir + '/' + simulation_name + '_b%d-'%bI
    field_file = data_path + 'field_energy_dot.gkyl'
    time_field_dot, field_dot_pb = read_dyn_vector(field_file)
    time_field, field = read_dyn_vector(data_path + 'field_energy.gkyl')
    field_dot_pb = field_dot_pb[1:]
    field = field[1:]
    #[ Add over blocks.
    if bI == 0:
      field_dot = field_dot_pb
    else:
      field_dot += field_dot_pb

    for sI, species_name in enumerate(species_names):
      data_path = data_dir + '/' + simulation_name + '_b%d-' + species_name
      file = data_path % bI 
  
      #[ Read data.
      time_fdot, fdot_pb = read_dyn_vector(file + '_fdot_integrated_moms.gkyl')
      time_distf, distf_pb = read_dyn_vector(file + '_integrated_moms.gkyl')

      source_file = file + '_source_integrated_moms.gkyl'
      has_source = does_file_exist(source_file)
      if has_source:
        time_src, src_pb = read_dyn_vector(source_file)

      nbflux = 0
      time_bflux, bflux_pb = list(), list()
      has_bflux = False
      for d in dirs:
        for e in edges:
          bflux_file = file + '_bflux_' + d + e + '_integrated_HamiltonianMoments.gkyl'
          has_bflux_at_boundary = does_file_exist(bflux_file)
          if has_bflux_at_boundary:
            time_bflux_tmp, bflux_pb_tmp = read_dyn_vector(bflux_file)
            time_bflux.append(time_bflux_tmp)
            bflux_pb.append(bflux_pb_tmp)
            has_bflux = has_bflux or has_bflux_at_boundary
            nbflux += 1

      #[ Select the Hamiltonian moment and remove the t=0 data point.
      fdot_pb = fdot_pb[1:,2]
      distf_pb = distf_pb[1:,2]

      if has_source:
        src_pb = src_pb[1:,2]
      else:
        src_pb = 0.0*fdot_pb

      if has_bflux:
        for i in range(nbflux):
          bflux_pb[i] = bflux_pb[i][1:,2]
      
        time_bflux_tot = time_bflux[0]
        bflux_pb_tot = bflux_pb[0] #[ Total boundary flux loss.
        for i in range(1,nbflux):
          bflux_pb_tot += bflux_pb[i]
      else:
        bflux_pb_tot = 0.0*fdot_pb

      #[ Add over blocks.
      if bI == 0:
        fdot = fdot_pb
        src = src_pb
        bflux_tot = bflux_pb_tot
        distf = distf_pb
      else:
        fdot += fdot_pb
        src += src_pb
        bflux_tot += bflux_pb_tot
        distf += distf_pb
  
  #[ Compute the relative error.
  mom_err = src - bflux_tot - (fdot -field_dot)
  mom_err_norm = mom_err*dt/(distf-field)

  #[ Plot the relative error.
  figProp2a = (7.5, 4.5)
  ax2aPos   = [0.11, 0.15, 0.87, 0.78]
  fig2a     = plt.figure(figsize=figProp2a)
  ax2a      = fig2a.add_axes(ax2aPos)

  hpl2a = list()
  hpl2a.append(ax2a.plot([-1.0,1.0], [0.0,0.0], color='grey', linestyle=':', linewidth=1))
  hpl2a.append(ax2a.plot(time_dt, mom_err_norm, color=defaultColors[0], linestyle=lineStyles[0], linewidth=2))

  ax2a.set_xlabel(r'Time ($s$)',fontsize=xyLabelFontSize, labelpad=+4)
  ax2a.set_ylabel(r'$E_{\dot{\mathcal{N}}}~\Delta t/\mathcal{N}$',fontsize=xyLabelFontSize, labelpad=0)
  ax2a.set_xlim( time_fdot[0], time_fdot[-1] )
  setTickFontSize(ax2a,tickFontSize)

  if save_fig_to_file:
    plt.savefig(fig_file_dir+simulation_name+'_particle_conservation_rel_error'+fig_file_format)
  else:
    plt.show()
