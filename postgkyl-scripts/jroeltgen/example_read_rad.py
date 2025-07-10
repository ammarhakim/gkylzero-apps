import read_radiation as read_rad
import numpy as np

rad_data=read_rad.gkyl_read_rad_fit_params("${HOME}/gkylzero/data/adas/radiation_fit_parameters.txt")
B0=2
vgrid=np.linspace(0,1e7,10)
mugrid=np.linspace(0,1e-16,10)
# Example is H+0 at ne=10^19 (and second includes nH0=10^19) 
rad_data.calc_drag(1,0,1e19,vgrid,mugrid,B0)
rad_data.print_cfl(1,0,1e19)
rad_data.print_cfl(1,0,1e19,density=1e19)
