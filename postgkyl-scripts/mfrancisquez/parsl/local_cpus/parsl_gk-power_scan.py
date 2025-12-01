#[ ....................................................... ]#
#[
#[ Run multiple Gkeyll GK simulations scanning input power
#[ using the parsl library.
#[
#[ Assumes this script is run from a directory that holds
#[ the Gkeyll executable. It creates a folder for each Gkeyll
#[ simulation to be run.
#[
#[ This runs each simulation on a CPU locally.
#[
#[ Manaure Francisquez.
#[ November 2025.
#[
#[ ....................................................... ]#

#[ Parsl library modules.
import parsl
from parsl.app.app import python_app, bash_app
from parsl.configs.local_threads import config
#[ File navigation/management modules.
import os
from os import path
import errno

gk_exec_name = "gk_sheath_2x2v_p1" #[ Simulation and executable name.
gk_exec_dir = os.getcwd() #[ Directory where Gkeyll executable is and where Parsl is launched from.

#[ Array of desired input powers.
input_power = [3.5e6+i*0.5e6 for i in range(2)]

#[ Each Gkeyll run will store its result in the folder
#[ names gk_exec_name-par#i$ where # is the parameter number (in case
#[ one scans multiple parameters, here we can only one, power) and
#[ $ is the index of the value of that parameter (e.g. here it'll
#[ run from 0 to len(input_power)-1).
run_folder_suffix = "par%di%d"

#[ ............... END OF USER INPUTS (maybe) ................ ]#

def check_mkdir(dirIn):
  #[ Check if folder 'dirIn' exists. If not, create it. The string
  #[ 'dirIn' must end with '/'.
  if not os.path.exists(os.path.dirname(dirIn)):
    try:
      os.makedirs(os.path.dirname(dirIn))
    except OSError as exc: # Guard against race condition
      if exc.errno != errno.EEXIST:
        raise

num_sims = len(input_power) #[ Number of simulations to run.
gk_exec_dir = gk_exec_dir + '/' #[ Added for safety.
gk_exec = gk_exec_dir + gk_exec_name #[ Gkeyll executable.

#[ Create parsl configuration.
parsl.load(config)

@bash_app
def gk_sim(stdout=(gk_exec_name+".out", "w"), stderr=(gk_exec_name+".err", "w"), sim_dir='./', power=4.5e6):
  #[ Command to run Gkeyll (as a string).
  os.chdir(sim_dir)
  gk_cmd = gk_exec + " -o Pin=" + str(power)
  return gk_cmd

#[ Run Gkeyll for each input parameter.
gk_status = [i for i in range(num_sims)]
for i in range(num_sims):
  pin = input_power[i] #[ Current parameter value.

  #[ Create and enter directory of current sim.
  gk_sim_dir = gk_exec + "-" + run_folder_suffix % (0,i) + "/"
  check_mkdir(gk_sim_dir)

  gk_status[i] = gk_sim(sim_dir = gk_sim_dir, power = pin)

#[ Wait for completion of all sims.
for i in range(num_sims):
  gk_status[i].result()

#[ Terminate parsl.
parsl.clear()
