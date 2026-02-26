"""
marimo_utils.py
Legacy wrapper script. Imports all modular components and exposes them 
to the Marimo dashboard notebook.
"""

# Import the actual logic from our new modular architecture
from config import FIELD_INFO, MP, ME, EV
from math_utils import calc_radial_correlation, get_2d_fluctuations
from data_loader import (
    load_datasets,
    process_simulation_run,
    load_2d_snapshot,
    load_integrated_moms,
    get_max_frames_and_time,
    process_spectra_for_sim
)
from plotting import (
    plot_saturation,
    plot_1d_profiles,
    plot_2d_comparison,
    plot_qpara_sol,
    plot_pdf_slice,
    plot_distf_slice,
    plot_parallel_mode_structure,
    plot_spectra_dashboard,
    plot_frequency_spectra,
    list_fields,
    scan_for_negativity
)

# (Optional) Print a quick verification that the wrapper loaded successfully
print("marimo_utils successfully loaded all submodules.")