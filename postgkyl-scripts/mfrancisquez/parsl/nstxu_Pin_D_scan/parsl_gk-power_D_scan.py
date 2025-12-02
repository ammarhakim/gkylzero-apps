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

import math
from itertools import product as it_product
#[ Parsl library modules.
import parsl
from parsl.app.app import python_app, bash_app
from parsl.config import Config
from parsl.providers import LocalProvider
from parsl.launchers import SimpleLauncher
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_interface
#[ File navigation/management modules.
import os
from os import path
import errno

gk_exec_name = "gk_multib_nstxu_DN_halfdomain_DandPscan_2x2v_p1" #[ Simulation and executable name.
gk_exec_dir = os.getcwd() #[ Directory where Gkeyll executable is and where Parsl is launched from.

#[ Scan parameters.
scan_param_values = [
  [1.5e6+i*0.5e6 for i in range(2)], #[ Power [MW].
  [0.2+i*0.2 for i in range(2)], #[ Particle diffusivity [m^2/s].
]

#[ Name of parameters in Gkeyll input file.
scan_param_names = [
  "Pin",
  "diffusionD",
]

#[ Each Gkeyll run will store its result in the folder
#[ names gk_exec_name-Pin#_diffD$ where # and $ are the indices of
#[ the input power and particle diffusivity used, relative to 
#[ scan_param_values[0] and scan_param_values[1], respectivelu.
run_folder_suffix = "Pin%d_diffD%d"

#[ SLURM parameters (it should be possible to query SLURM from Python, but haven't figured out how yet).
gpus_per_sim = 8

#[ ............... END OF USER INPUTS (maybe) ................ ]#

def get_env_var(var_name, default_value):
  #[ Read the value of environment variable 'var_name'.
  #[ If not found set it to 'default_value'.
  var_value = os.environ.get(var_name)
  if var_value:
    print(f"Slurm {var_name}: {var_value}")
  else:
    var_value = default_value
    print(f"{var_name} not found. Setting to: {var_value}")

  return var_value

def check_mkdir(dirIn):
  #[ Check if folder 'dirIn' exists. If not, create it. The string
  #[ 'dirIn' must end with '/'.
  if not os.path.exists(os.path.dirname(dirIn)):
    try:
      os.makedirs(os.path.dirname(dirIn))
    except OSError as exc: # Guard against race condition
      if exc.errno != errno.EEXIST:
        raise

def flatten_list(nested_list):
  #[ Flatten a list of lists. Call as list(flatten_list(list)).
  for item in nested_list:
    if isinstance(item, list):
      yield from flatten_list(item)
    else:
      yield item

num_nodes         = int(get_env_var('SLURM_JOB_NUM_NODES', 1))
cores_per_node    = int(get_env_var('SLURM_CPUS_ON_NODE', 128)) #[ Perlmutter: 64 cores per node, 2 threads each.
num_gpus_per_node = 4 #[ Perlmutter: 4 gpus per node.
num_gpus          = int(get_env_var('SLURM_GPUS', num_gpus_per_node))
nodes_per_sim     = math.ceil(gpus_per_sim/num_gpus_per_node)

#[ Create a list of parameters for each sim.
sim_params = [scan_param_values[0][i] for i in range(len(scan_param_values[0]))]
for i in range(1,len(scan_param_names)):
  curr_prod = list(it_product(sim_params, scan_param_values[i]))
  sim_params = [list(flatten_list(list(curr_prod[j]))) for j in range(len(curr_prod))]

num_sims      = len(sim_params) #[ Number of simulations to run.
sims_per_node = num_sims/num_nodes
cpus_per_task = int(cores_per_node/(sims_per_node*gpus_per_sim)) #[ Perlmutter's -c argument.

#[ Ensure that job has enough resources.
if not (num_sims*gpus_per_sim <= num_gpus):
  raise ValueError(f"The number of simulations to be launched ({num_sims}) times the number of GPUs per simulation ({gpus_per_sim}) must be equal to or less than the number of GPUs allocated for this job ({num_gpus}).")

gk_exec_dir = gk_exec_dir + '/' #[ Added for safety.
gk_exec     = gk_exec_dir + gk_exec_name #[ Gkeyll executable.

#[ Create parsl configuration.
config = Config(
  executors = [
    HighThroughputExecutor(
      label = 'perlmuter_HTEX',
      max_workers_per_node = num_sims,
      provider = LocalProvider(
        nodes_per_block = num_nodes,
        launcher = SimpleLauncher(),
      ),
    )
  ],
  strategy = 'simple',
)
parsl.load(config)

@bash_app
def gk_sim(stdout = (gk_exec_name+".out", "w"), stderr = (gk_exec_name+".err", "w"),
           sim_dir = './', exec_full_path = './'+gk_exec_name,
           nodes_per_sim = 1, ranks_per_sim = 1, cpus_per_task = 1,
           sim_par_values = [4.5e6, 0.22], sim_par_names = ["Pin","diffusionD"]):
  #[ Go to directory for this sim.
  import os
  os.chdir(sim_dir)

  #[ Command to run Gkeyll (as a string).
  gk_cmd = 'srun -u -N ' + str(nodes_per_sim) + ' -n ' + str(ranks_per_sim) + ' --gpus ' + str(ranks_per_sim) + \
    ' -c '+str(cpus_per_task) + ' --cpu_bind=cores --exclusive ' + \
    exec_full_path + " -g -M -d " + str(ranks_per_sim) + " -o " + \
    sim_par_names[0] + "=" + str(sim_par_values[0]) + "," + \
    sim_par_names[1] + "=" + str(sim_par_values[1]) + " -s 2"
  return gk_cmd

#[ Run Gkeyll for each input parameter.
gk_status = [i for i in range(num_sims)]
for i in range(num_sims):
  params = sim_params[i] #[ Current parameter values.

  #[ Create and enter directory of current sim.
  gk_sim_dir = gk_exec + "-" + \
    run_folder_suffix % (scan_param_values[0].index(params[0]),scan_param_values[1].index(params[1])) + "/"
  check_mkdir(gk_sim_dir)

  #[ Run sim.
  gk_status[i] = gk_sim(
    sim_dir = gk_sim_dir,
    exec_full_path = gk_exec,
    nodes_per_sim = nodes_per_sim,
    ranks_per_sim = gpus_per_sim,
    cpus_per_task = cpus_per_task,
    sim_par_values = params,
    sim_par_names = scan_param_names,
  )

#[ Wait for completion of all sims.
for i in range(num_sims):
  gk_status[i].result()

#[ Terminate parsl.
parsl.clear()
