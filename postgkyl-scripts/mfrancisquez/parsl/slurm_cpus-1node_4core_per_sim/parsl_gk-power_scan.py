#[ ....................................................... ]#
#[
#[ Run multiple Gkeyll GK simulations scanning input power
#[ using the parsl library.
#[
#[ Assumes this script is run from a directory that holds
#[ the Gkeyll executable. It creates a folder for each
#[ Gkeyll simulation to be run.
#[
#[ This runs each simulation on a 4 CPU cores within a
#[ single-node Slurm job.
#[
#[ Manaure Francisquez.
#[ November 2025.
#[
#[ ....................................................... ]#

#[ Parsl library modules.
import parsl
from parsl.app.app import python_app, bash_app
from parsl.config import Config
from parsl.providers import LocalProvider
from parsl.launchers import SrunLauncher
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_interface
#[ File navigation/management modules.
import os
from os import path
import errno

gk_exec_name = "gk_sheath_2x2v_p1" #[ Simulation and executable name.
gk_exec_dir = os.getcwd() #[ Directory where Gkeyll executable is and where Parsl is launched from.

#[ Array of desired input powers.
input_power = [3.5e6+i*0.5e6 for i in range(6)]

#[ Each Gkeyll run will store its result in the folder
#[ names gk_exec_name-par#i$ where # is the parameter number (in case
#[ one scans multiple parameters, here we can only one, power) and
#[ $ is the index of the value of that parameter (e.g. here it'll
#[ run from 0 to len(input_power)-1).
run_folder_suffix = "par%di%d"

#[ SLURM parameters (it should be possible to query SLURM from Python, but haven't figured out how yet).
num_nodes = 1
cores_per_node = 128
cores_per_sim = 4

max_wallclock_time_per_sim = 250

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
config = Config(
  executors=[
    HighThroughputExecutor(
      label='PM_HTEX_headless',
      # one worker per manager / node
      max_workers_per_node=int(cores_per_node/cores_per_sim),
      provider=LocalProvider(
        nodes_per_block=num_nodes,
        launcher=SrunLauncher(overrides='-c '+str(cores_per_sim)+' --cpu_bind=cores'),
        cmd_timeout=max_wallclock_time_per_sim,
        init_blocks=1,
        max_blocks=1,
      ),
    )
  ],
  strategy=None,
)
parsl.load(config)

@bash_app
def gk_sim(stdout=(gk_exec_name+".out", "w"), stderr=(gk_exec_name+".err", "w"),
           sim_dir='./', exec_full_path = './'+gk_exec_name,
           mpi_processes=1, power=4.5e6):
  #[ Go to directory for this sim.
  import os
  os.chdir(sim_dir)

  #[ Command to run Gkeyll (as a string).
  gk_cmd = exec_full_path + "-M -c " + str(mpi_processes) + " -o Pin=" + str(power)
  return gk_cmd

#[ Run Gkeyll for each input parameter.
gk_status = [i for i in range(num_sims)]
for i in range(num_sims):
  pin = input_power[i] #[ Current parameter value.

  #[ Create and enter directory of current sim.
  gk_sim_dir = gk_exec + "-" + run_folder_suffix % (0,i) + "/"
  check_mkdir(gk_sim_dir)

  #[ Run sim.
  gk_status[i] = gk_sim(
    sim_dir = gk_sim_dir,
    exec_full_path = gk_exec,
    mpi_processes = cores_per_sim,
    power = pin
  )

#[ Wait for completion of all sims.
for i in range(num_sims):
  gk_status[i].result()

#[ Terminate parsl.
parsl.clear()
