Using Parsl to run Gkeyll
-------------------------

Examples of using the Python Parsl library to run batches of Gkeyll simulations.
We use a power scan in a modified version of `rt_gk_sheath_2x2v_p1` as an example,
and the scripts below show different ways of using Parsl to do this power scan.

- `local_cpus`: Each sim on a local CPU.
- `slurm_cpus-1node_1core_per_sim`: Launch a slurm job requesting 1 node, and run each sim on a 1 CPU. 
- `slurm_cpus-1node_4core_per_sim`: Launch a slurm job requesting 1 node, and run each sim on a 4 CPUs. 
