Using Parsl to run Gkeyll
-------------------------

Examples of using the Python Parsl library to run batches of Gkeyll simulations.
We use a power scan in a modified version of `rt_gk_sheath_2x2v_p1` as an example,
and the scripts below show different ways of using Parsl to do this power scan.

NOTE: the following folders also reflect my learning curve. I would revise the earlier
folders and try to do them with scripts like those in the last folder.

- `local_cpus`: Each sim on a local CPU.
- `slurm_cpu-1node_1core_per_sim`: request 1 node, run each sim on a 1 CPU. 
- `slurm_cpu-1node_4core_per_sim`: request 1 node, run each sim on a 4 CPUs. 
- `slurm_gpu-1node_2gpu_per_sim`: request 1 node, run 2 sims on 2 GPUs each. 
- `slurm_gpu-2node_2gpu_per_sim`: request 2 node, run 4 sims on 2 GPUs each. 
- `slurm_gpu-2node_4gpu_per_sim`: request 2 node, run 2 sims on 4 GPUs each. 
- `slurm_gpu-4node_8gpu_per_sim`: request 4 node, run 2 sims on 8 GPUs each. 

More realistic cases:
- `nstxu_Pin_D_scan`: NSTX U power and diffusion scan.

