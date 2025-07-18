import numpy as np
from scipy.integrate import quad
from scipy.special import erf
#[ Append postgkyl wrappers.
from scipy import special
from scipy import optimize


# Calculate the Pastukhov potential confinement time from an adiabatic electron simulation

R = 1.733796e+01 / 9.992183e-01 # Read from postgkyl
n0 = 1.7127e19 # Read from the simulation at midplane
nu_ii = 4.878278e+00 # Read from the simulation at midplane

eps0, mu0 = 8.8541878176204e-12, 1.2566370614359e-06
eV        = 1.602176487e-19
qe, qi    = -1.602176487e-19, 1.602176487e-19
me, mp    = 9.10938215e-31, 1.672621637e-27

mi        = 2.014*mp                         #[ Deuterium ion mass.
Te0       = 940*eV

#[ Electron-electron collision freq.
logLambdaElc = 6.6 - 0.5*np.log(n0/1e20) + 1.5*np.log(Te0/eV)
nuElc        = logLambdaElc*(eV**4)*n0/(6*np.sqrt(2)*(np.pi**(3/2))*(eps0**2)*np.sqrt(me)*(Te0**(3/2)))


def tau_pe(x,R,frac):
  #[ Electron confinement time as a function of x=e*phi/Te and mirror ratio R.
  #[ This has a 1/4 (Cohen) instead of a 1/2 (Pastukhov). However, Cohen assumes
  #[ ee and ei collisions. If we want to consider only ee collisions use this
  #[ function with frac=0.5 (essentially turning the 1/4 into 1/2).
  def G(R):
    return ((2.*R+1)/(2.*R))*np.log(4.*R+2)
  def I_x(x):
    return 1.+0.5*np.sqrt(np.pi*x)*np.exp(1./x)*special.erfc(np.sqrt(1./x))
  return (np.sqrt(np.pi)/4.)*(1./(frac*nuElc))*G(R)*x*np.exp(x)/I_x(1./x)

# Compute analytical estimate from Najmabadi. It's more easily implemented from Post 1987
def Najmabadi_confinement_time(P, R, ZpFl=1, coeff = 0.84):
    w_term = np.sqrt(1 + 1/(R*(ZpFl - 1/(4*P)))) #Is this 1/4P or P/4
    u_eff  = P + np.log(w_term) #same as Najmabadi's a**2

    integrandNaj = lambda t: np.exp(-t) / t
    I_term, error = quad(integrandNaj, u_eff, np.inf)
    I_term = (ZpFl + 1/4)*u_eff*np.exp(u_eff)*I_term - 1/4 

    Loss_Najmabadi = 1/nuElc * \
        np.sqrt(np.pi)/4 * u_eff*np.exp(u_eff)/I_term \
        * (np.log((w_term+1)/(w_term-1)) - coeff)#0.84
    
    return Loss_Najmabadi


def Rosen_Dougherty_confinement_time(P, R, ZpFl, coeff = 0):
    w_term = np.sqrt(1 + 2*P/(R*ZpFl)) 
    a_term  = np.sqrt(P + np.log(w_term)) 

    Loss_Rosen = 1/nuElc * \
        1/(2*ZpFl / (np.log((w_term+1)/(w_term-1)) - coeff) * (1-erf(a_term)))
    
    return Loss_Rosen

def tau_pi(R):
  #[ Ion confinement time as a function of mirror ratio R.
  tau_i = 1/nu_ii
  return tau_i*np.log10(R)

def rootEq(x,R):
  return tau_pi(R)-Rosen_Dougherty_confinement_time(x,R,1,1.065)

ephi_over_Te = optimize.ridder(rootEq, 1, 20, args=(R))

print(f"e*phi/Te = {ephi_over_Te:.3f}")