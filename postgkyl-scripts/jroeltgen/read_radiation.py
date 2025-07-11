import os
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as syc
from matplotlib.pyplot import cm

# Author: J. Roeltgen

# Class containing a single fit
class rad_fit_parameters:
    def __init__(self, ne, A, alpha, beta, V0, gamma, te_intervals, te, Lz):
        self.electron_density=ne
        self.A=A
        self.alpha=alpha
        self.beta=beta
        self.V0=V0
        self.gamma=gamma
        self.te_intervals=te_intervals
        self.te=te
        self.Lz=Lz
        self.c_const = 8*syc.elementary_charge/np.sqrt(syc.pi)*(2*syc.elementary_charge/syc.m_e)**(gamma/2)
        self.vpar_nup = np.linspace(-2e7,2e7,1000)
        self.mu_nup = np.linspace(0,syc.m_e*self.vpar_nup[-1]**2/2,1000)
        self.vpar_nupp = np.linspace(-2e7,2e7,1000)
        self.mu_nupp = np.linspace(0,syc.m_e*self.vpar_nupp[-1]**2/2,1000)
        self.nuprime=[]
        self.nudoubleprime=[]

    def calc_nu(self, vpar, mu, B):
        vmag = np.zeros((len(vpar),len(mu)))
        for i,mu2 in enumerate(mu):
            vmag[:,i] = np.sqrt(vpar**2+2*B*mu2/syc.m_e)
        V0 = self.V0*np.sqrt(2*syc.elementary_charge/syc.m_e)
        x = vmag/V0
        mask = vmag==0
        nu = self.A*(self.alpha+self.beta)/(self.beta*(x**-self.alpha)+self.alpha*x**self.beta)*vmag**self.gamma
        nu[mask]=0
        return nu/self.c_const

    def calc_nuprime(self, vpar=None,  mu=None, B=1):
        if (vpar is None):
            vpar = self.vpar_nup
        if (mu is None):
            mu = self.mu_nup
        self.vpar_nup=vpar
        self.mu_nup=mu
        self.nuprime = np.zeros((len(vpar),len(mu)))
        nu = self.calc_nu(vpar,mu,B)
        for i,v in enumerate(vpar):
            self.nuprime[i,:] = nu[i,:]*v
        
    def calc_nudoubleprime(self, vpar=None, mu=None, B=1):
        if (vpar is None):
            vpar = self.vpar_nupp
        if (mu is None):
            mu = self.mu_nupp
        self.vpar_nupp=vpar
        self.mu_nupp=mu
        self.nudoubleprime = np.zeros((len(vpar),len(mu)))
        nu = self.calc_nu(vpar,mu,B)
        for i,mu2 in enumerate(mu):
            self.nudoubleprime[:,i] = 2*nu[:,i]*mu2

    # density - radiating species density
    def print_cfl(self, porder=[2,1], pdim=4, density=None):
        cflv = np.zeros((self.nuprime.shape))
        cflmu = cflv
        if(len(self.nuprime)==0):
            self.calc_nuprime()
        if(len(self.nudoubleprime)==0):
            self.calc_doubleprime()
        for i,v in enumerate(self.vpar_nup[0:-1]):
            rdv2 = 2/(self.vpar_nup[i+1]-v)
            cflv[i,:] = (self.nuprime[i,:])*(0.5*(2+porder[0]+1)*rdv2*2**(-0.5*(pdim-1)))
        for i,mu in enumerate(self.mu_nupp[0:-1]):
            rdv2 = 2/(self.mu_nupp[i+1]-mu)
            cflmu[:,i] = (self.nudoubleprime[:,i])*(0.5*(2+porder[1]+1)*rdv2*2**(-0.5*(pdim-1)))
        cfl = cflv+cflmu
        if(density is None):
            print("cfl frequency divided by impurity density = %r" % (np.amax(cfl)) )
        else:
            print("cfl frequency at %r m^-3 = %e" % (density, np.amax(cfl)*density) )
            
# Class denoting a single charge state of a single element
class radiating_state:
    def __init__(self, atomic_number, charge_state, state_exists, number_of_densities, rad_fits):
        self.atomic_number=atomic_number
        self.charge_state=charge_state
        self.state_exists=state_exists
        self.number_of_densities=number_of_densities
        self.rad_fits=rad_fits
        self.electron_densities=[]

    def set_state_exists(self, data):
        self.state_exists = data

    def set_num_densities(self, data):
        self.number_of_densities = data

    def set_electron_densities(self, data):
        self.electron_densities.append(10**data)
        
    def add_rad_fit(self, data):
        self.rad_fits.append(data)

    def get_ne_idx(self, electron_density):
        return np.abs(np.asarray(self.electron_densities)-electron_density).argmin()        

# Class containing all states for an element
class element_states:
    def __init__(self, max_charge_state):
        self.max_charge_state = max_charge_state
        self.all_states=[]
        self.existing_charge_states=[]
        
        for j in range(max_charge_state+1):
            self.all_states.append(radiating_state(max_charge_state+1, j, False, 0, []))

    def __getitem__(self, idx):
        return self.all_states[idx]

# Class containing all elements/charge states in given file
class all_radiation_states:
    def __init__(self, max_atomic_number):
        self.max_atomic_number = max_atomic_number
        self.all_elements=[element_states(-1)]
        self.existing_z=[]
        
        for zm1 in range(max_atomic_number+1):
            self.all_elements.append(element_states(zm1))

    def __getitem__(self, idx):
        return self.all_elements[idx]

    # return array of electron densities for given element and charge state
    def get_electron_densities(self, atomic_number, charge_state):
        return self.all_elements[atomic_number].all_states[charge_state].electron_densities

    # return available atomic numbers
    def get_atomic_nums(self):
        return self.existing_z

    def get_available_charge_states(self, atomic_number):
        return self.all_elements[atomic_number].existing_charge_states

    # Return the nu' portion of the drag (part acting in vpar direction)
    def get_nuprime(self, atomic_number, charge_state, electron_density):
        idx = self.all_elements[atomic_number].all_states[charge_state].get_ne_idx(electron_density)
        fit = self.all_elements[atomic_number].all_states[charge_state].rad_fits[idx]
        if(len(fit.nuprime)==0):
            print("Calculating nuprime with default velocity grid.")
            fit.calc_nuprime()
        return fit.nuprime, fit.vpar_nup, fit.mu_nup

    # Return the nu'' portion of the drag (part acting in mu direction)
    def get_nudoubleprime(self, atomic_number, charge_state, electron_density):
        idx = self.all_elements[atomic_number].all_states[charge_state].get_ne_idx(electron_density)
        fit = self.all_elements[atomic_number].all_states[charge_state].rad_fits[idx]
        if(len(fit.nudoubleprime)==0):
            print("Calculating nudoubleprime with default velocity grid.")
            fit.calc_nudoubleprime()
        return fit.nudoubleprime, fit.vpar_nupp, fit.mu_nupp

    # Print fit parameters for given fit
    def print_fit_params(self, atomic_number, charge_state, electron_density):
        idx = self.all_elements[atomic_number].all_states[charge_state].get_ne_idx(electron_density)
        fit = self.all_elements[atomic_number].all_states[charge_state].rad_fits[idx]
        print("A=%r, alpha=%r, beta=%r, gamma=%r, V0=%r" % (fit.A, fit.alpha, fit.beta, fit.gamma, fit.V0) )

    # Calculate and store drag terms. vpar and mu are 1D arrays of the velocity grids and B is the
    # magnetic field
    def calc_drag(self, atomic_number, charge_state, electron_density, vpar=None, mu=None, B=1):
        idx = self.all_elements[atomic_number].all_states[charge_state].get_ne_idx(electron_density)
        fit = self.all_elements[atomic_number].all_states[charge_state].rad_fits[idx]
        fit.calc_nuprime(vpar, mu, B)
        fit.calc_nudoubleprime(vpar, mu, B)

    # Print cfl
    # porder - polynomial order
    # pdim - phase dimensions
    # density - radiating species density (if not entered cfl is cfl/density)
    def print_cfl(self, atomic_number, charge_state, electron_density, porder=[2,1], pdim=4, density=None):
        idx = self.all_elements[atomic_number].all_states[charge_state].get_ne_idx(electron_density)
        fit = self.all_elements[atomic_number].all_states[charge_state].rad_fits[idx]
        fit.print_cfl(porder,pdim,density)

    def plot_emis(self, atomic_number, charge_state, electron_density=1e19, options={}):
        self.plot_emis_helper([atomic_number], [charge_state], [electron_density], options=options)

    def plot_emis_all_elements(self, charge_state, electron_density=1e19, options={}):
        z=self.get_atomic_nums()
        for i in range(charge_state):
            if i in z:
                z.remove(i)
        self.plot_emis_compare(z, [charge_state], [electron_density], options)

    def plot_emis_all_states(self, atomic_number, electron_density=1e19, options={}):
        cstates = self.get_available_charge_states(atomic_number)
        self.plot_emis_compare([atomic_number], cstates, [electron_density], options)

    def plot_emis_all_ne(self, atomic_number, charge_state, options={}):
        ne = self.get_electron_densities(atomic_number, charge_state)
        self.plot_emis_compare([atomic_number], [charge_state], ne, options)
        
    # Plots tensor product of atomic_number, charge_state and electron_density lists.
    # So, atomic_number={2,3,18}, charge_state={0,1}, electron_density=1e19
    # would plot 6 curves: charge states 0 and 1 for all three elements z=(2,3,18).
    # Therefor a restriction is maximum charge_state<minimum atomic_number
    def plot_emis_compare(self, atomic_number, charge_state, electron_density=1e19, options={}):
        if isinstance(atomic_number, int):
            z = list({atomic_number})
        else:
            z = atomic_number
        if isinstance(charge_state, int):
            charge = list({charge_state})
        else:
            charge = charge_state
        if isinstance(electron_density, float):
            ne = list({electron_density})
        else:
            ne = electron_density

        markers=[".","o","v",">","s","*","+","x","d","|","X","P","^","<","p"]    
        linestyle_tuple = [
            ('solid',                 (0, ())),
            ('loosely dotted',        (0, (1, 10))),
            ('dotted',                (0, (1, 1))),
            ('densely dotted',        (0, (1, 1))),
            ('long dash with offset', (5, (10, 3))),
            ('loosely dashed',        (0, (5, 10))),
            ('dashed',                (0, (5, 5))),
            ('densely dashed',        (0, (5, 1))),
            
            ('loosely dashdotted',    (0, (3, 10, 1, 10))),
            ('dashdotted',            (0, (3, 5, 1, 5))),
            ('densely dashdotted',    (0, (3, 1, 1, 1))),
            
            ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
            ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
            ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
        lstyles=[]
        allc=[]
        allmark=[]
        atomic_z=[]
        cstate=[]
        allne=[]
        for i in range(len(z)):
            ltuple = linestyle_tuple[i]
            colors = iter(cm.rainbow(np.linspace(0, 1, len(charge) )))
            for j in range(len(charge)):
                c = next(colors)
                for k in range(len(ne)):
                    lstyles.append(ltuple[1])
                    allc.append(c)
                    allmark.append(markers[k])
                    atomic_z.append(z[i])
                    cstate.append(charge[j])
                    allne.append(ne[k])
        self.plot_emis_helper(atomic_z, cstate, allne, allc, lstyles, allmark, options=options)


    #Basic emissivity plotting function. Takes lists of atomic
        #number, charge state, and electron_density 
    def plot_emis_helper(self, atomic_number, charge_state, electron_density, colors = ["black"], linestyles=["-"], markers=["."], options={}):
        while len(linestyles)<len(atomic_number):
            linestyles.append(linestyles)
        if len(colors)<len(atomic_number):
            colors.append(colors)
        while len(markers)<len(atomic_number):
            markers.append(markers)

        for i in range(len(atomic_number)):
            idx = self.all_elements[atomic_number[i]].all_states[charge_state[i]].get_ne_idx(electron_density[i])
            fit = self.all_elements[atomic_number[i]].all_states[charge_state[i]].rad_fits[idx]
            key = "$Z="+str(atomic_number[i])+"\ Charge="+str(charge_state[i])+"\ ne="+str(electron_density[i])+"$"
            plt.loglog(fit.te, fit.Lz, color=colors[i], marker=markers[i], linestyle=linestyles[i], label=key, **options)
            
        plt.xlabel("T_e (eV)")
        plt.ylabel("Emissivity (Wm^3)")
        plt.legend(loc="best")
        plt.yscale('log')
        plt.xscale('log')
        plt.show()        


def read_two_numbers(fptr):
    str_line = fptr.readline().strip()
    if str_line:
        delim = '\=|,|;|:'
        parts = re.split(delim,str_line)
        num1 = int(parts[1].strip())
        num2 = int(parts[3].strip())
    return num1, num2

# Read in radiation fit parameters from given file and return all_radiation_states object
def gkyl_read_rad_fit_params(filepath = os.path.join(os.environ.get('GKYL_SHARE_DIR', ''), 'adas', 'radiation_fit_parameters.txt')):    
    with open(filepath, 'r') as fptr:
        # Read header: Max z of elements, number of elements
        max_atomic_number, number_elements = read_two_numbers(fptr)
        max_charge_state = max_atomic_number
        rad_data = all_radiation_states(max_atomic_number)
        
        for i in range(number_elements):
            # For each element, read atomic number and # of charge states
            atomic_number, num_of_charge_states = read_two_numbers(fptr)
            rad_data.existing_z.append(atomic_number)
            
            #atomic_number -= 1
            for j in range(num_of_charge_states):
                # For each charge state, read # of density intervals
                charge_state, density_intervals = read_two_numbers(fptr)
                charge_state -= 1  # convert to zero-based array
                rad_data.all_elements[atomic_number].existing_charge_states.append(charge_state)                
                if density_intervals > 0:
                    rad_data.all_elements[atomic_number].all_states[charge_state].set_num_densities(density_intervals)
                    rad_data.all_elements[atomic_number].all_states[charge_state].set_state_exists(True)
                    ne = np.zeros(density_intervals)
                                                             
                    for k in range(density_intervals):
                        line = fptr.readline()
                        if line is None:
                            density_intervals -= 1
                            rad_data.all_elements[atomic_number].all_states[charge_state].set_num_densities(density_intervals)
                        else:
                            data = line.split()
                            te_intervals = int(data[6])
                            te_data = list(map(float, fptr.readline().split()))                 
                            lz_data = list(map(float, fptr.readline().split()))
                            rad_data.all_elements[atomic_number].all_states[charge_state].set_electron_densities(float(data[0]))
                            rad_data.all_elements[atomic_number].all_states[charge_state].add_rad_fit(rad_fit_parameters(float(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5]), te_intervals, te_data, lz_data))
                            
    return rad_data


# Example usage:
# import read_radiation as read_rad
# rad_data = read_rad.gkyl_read_rad_fit_params('$HOME/gkylzero/data/adas/radiation_fit_parameters.txt')

# Plot argon +2 emissivity in red
# options = {'color':'red} --  dictionary of matplotlib.pyplot options
# atomic_number = 18
# charge_state = 2
# electron_density = 1e19 -- precise number is irrelevant for argon
# rad_data.plot_emis(atomic_number, charge_state, electron_density, options)
