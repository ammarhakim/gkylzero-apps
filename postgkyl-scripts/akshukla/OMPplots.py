import numpy as np
import matplotlib.pyplot as plt

plt.figure()
xplot = np.r_[mom_data_list[2]["Ri"][:, zidx[2]],mom_data_list[10]["Ri"][:, zidx[10]]]
yplot = np.r_[mom_data_list[2]["ionTemp"][:, zidx[2]],mom_data_list[10]["ionTemp"][:, zidx[10]]]
plt.plot(xplot, yplot)
plt.ylim(0,1e4)
plt.axvline(x=mom_data_list[2]["Ri"][-1, zidx[2]], color='grey', linestyle='dashed')
plt.xlabel('R [m]', fontsize=20)
plt.ylabel(r'$T_{D^+}$' + ' [eV]', fontsize=20)
#plt.title("Ion Temperature At OMP", fontsize=20)
plt.tight_layout()
plt.figure()
xplot = np.r_[mom_data_list[2]["Ri"][:, zidx[2]],mom_data_list[10]["Ri"][:, zidx[10]]]
yplot = np.r_[mom_data_list[2]["elcM0"][:, zidx[2]],mom_data_list[10]["elcM0"][:, zidx[10]]]
plt.plot(xplot, yplot)
plt.ylim(0,1.75e18)
plt.axvline(x=mom_data_list[2]["Ri"][-1, zidx[2]], color='grey', linestyle='dashed')
plt.xlabel('R [m]', fontsize=20)
plt.ylabel(r'$n_e\, [\mathrm{m}^{-3}]$', fontsize=20)
#plt.title("Density At OMP", fontsize=20)
plt.tight_layout()
