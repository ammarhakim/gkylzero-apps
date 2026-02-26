import marimo

__generated_with = "0.19.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import sys
    import numpy as np
    import matplotlib.pyplot as plt

    # 1. Define the path
    utils_path = os.path.expanduser('~/gkylzero-apps/postgkyl-scripts/tbernard/')

    # 2. CRITICAL STEP: Actually append it to the system path
    if utils_path not in sys.path:
        sys.path.append(utils_path)

    # 3. Print to confirm where we are looking (helps debugging)
    print(f"Looking for custom libs in: {utils_path}")

    # 4. Import the library
    try:
        import marimo_utils as mlib
        # Force a reload in case you edited the file but the kernel cached the old broken version
        import importlib
        importlib.reload(mlib)
        print("Library imported successfully.")
    except ImportError as e:
        print(f"Error importing library: {e}")
    return mlib, mo, np, plt


@app.cell
def _(mo):
    mo.md("""
    # Gkeyll Simulation Comparator
    **Reactive Analysis Dashboard**
    """)
    return


@app.cell
def _():
    sim_definitions = {
        '0.5MW med q': {
        'path' : '/pscratch/sd/t/tnbernar/turb-spread/0.5_MW_qmed/',
        'color': 'black', 'ls' : '-'},
        '1.5MW med q': {
       'path' : '/pscratch/sd/t/tnbernar/turb-spread/1.5_MW_qmed/',
        'color': 'blue', 'ls' : '-'},
        '2.5MW med q': {
        'path' : '/pscratch/sd/t/tnbernar/turb-spread/2.5_MW_qmed/',
        'color': 'red', 'ls' : '-'},
    # q_axis + (q_sep - q_axis)*pow(r/a_mid, 2.0);
       # '1.5MW low q': { #q_sep = 4
       # 'path' : '/pscratch/sd/t/tnbernar/turb-spread/1.5_MW_qlow/',
       # 'color': 'C0', 'ls' : '-'},
       # '1.5MW mid q': { #q_sep = 6
       # 'path' : '/pscratch/sd/t/tnbernar/turb-spread/1.5_MW_qmed/',
       # 'color': 'C1', 'ls' : '-'},
       # '1.5MW high q': { #q_sep = 8
       # 'path' : '/pscratch/sd/t/tnbernar/turb-spread/1.5_MW_qhigh/',
       # 'color': 'C2', 'ls' : '-'},
       # 'PT': {
       # 'path' : '/pscratch/sd/t/tnbernar/ceda-data/d3d/posD-3x/new-gkeyll/restart-hires-neut/', 
       # 'color': 'red', 'ls' : '-'},
       #  'NT': {
       # 'path' : '/pscratch/sd/t/tnbernar/ceda-data/d3d/negD-3x/new-gkeyll/restart-hires-neut/',
       # 'color': 'blue', 'ls' : '-'},
    }
    return (sim_definitions,)


@app.cell
def _(mo):
    # 1. Create your UI elements first
    fstart_input = mo.ui.number(value=80, label="Start Frame")
    fend_input = mo.ui.number(value=100, label="End Frame")
    step_input = mo.ui.number(value=1, label="Frame Step")
    amu_input = mo.ui.number(value=2.014, label="AMU (Mass)")
    lcfs_input = mo.ui.number(value=0.1, label="LCFS Shift (m)")

    # 2. Group them into a vertical stack (optional, but looks better)
    settings_stack = mo.vstack([
        mo.md("### ⚙️ Analysis Settings"),
        fstart_input,
        fend_input,
        step_input,
        amu_input,
        lcfs_input
    ])

    # 3. Pass the stack to the sidebar function
    mo.sidebar(settings_stack)
    return amu_input, fend_input, fstart_input, lcfs_input


@app.cell
def _(mo):
    mo.md("### Machine Geometry Configuration")

    r_axis_input = mo.ui.number(value=1.65, step=0.01, label="Magnetic Axis (R_axis) [m]")
    r_lcfs_input = mo.ui.number(value=2.17, step=0.01, label="Separatrix (R_LCFS) [m]")

    # NEW: Magnetic Field Input
    b_axis_input = mo.ui.number(value=2.0, step=0.1, label="B_axis [T]")

    geometry_ui = mo.hstack([r_axis_input, r_lcfs_input, b_axis_input], justify="start")
    return b_axis_input, r_axis_input, r_lcfs_input


@app.cell
def _(
    amu_input,
    b_axis_input,
    fend_input,
    fstart_input,
    mlib,
    mo,
    sim_definitions,
):
    @mo.cache # simple caching to prevent re-runs on minor UI tweaks if inputs match
    def load_data(sims, start, end, mass, b_axis):
        # This calls the batch loader we added to the library
        return mlib.load_datasets(sims, fstart=start, fend=end, step=1, amu=mass, B_axis=b_axis)

    with mo.status.spinner(title="Processing Gkeyll Data..."):
        # Load the data based on current UI values
        comparison_data = load_data(
            sim_definitions,
            fstart_input.value,
            fend_input.value,
            amu_input.value,
            b_axis_input.value
        )

    mo.md(f"✅ **Loaded {len(comparison_data)} simulations.**")
    return (comparison_data,)


@app.cell
def _(mlib, mo):
    mo.md("## 1D Profiles & Statistics")

    # Physical geometry inputs to calculate the minor radius scaling (Defaulted to DIII-D)
    #r_axis_input = mo.ui.number(value=1.65, step=0.01, label="Physical R_axis [m]")
    #r_lcfs_input = mo.ui.number(value=2.17, step=0.01, label="Physical R_LCFS [m]")

    trim_input = mo.ui.number(value=5, step=1, start=0, label="Trim Boundary Cells (x_pts)")

    # 1. Define any fields you want to HIDE from the dropdown
    hidden_fields = [
        'QparaAve','Lc_ave',
        'int_n', 'int_en', 'int_nvz'
    ]

    # 2. Filter the available fields
    available_1d_fields = [
        key for key in mlib.FIELD_INFO.keys() 
        if key not in hidden_fields
    ]

    field_selector = mo.ui.multiselect(
        options=available_1d_fields,
        value=['neAve', 'I_flux', 'gamma_mhd'],
        label="Select Fields to Plot:"
    )
    return field_selector, trim_input


@app.cell
def _(
    comparison_data,
    field_selector,
    lcfs_input,
    mlib,
    mo,
    r_axis_input,
    r_lcfs_input,
):
    # Stack the UI
    _controls = mo.vstack([
        #mo.hstack([r_axis_input, r_lcfs_input]),
        field_selector
    ])

    if comparison_data and field_selector.value:
        # Generate the plot
        _fig_1d = mlib.plot_1d_profiles(
            comparison_data, 
            field_selector.value,
            lcfs_shift=lcfs_input.value,   # Your existing grid shift
            r_lcfs=r_lcfs_input.value,     # For calculating minor radius
            r_axis=r_axis_input.value,      # For calculating minor radius
            trim_pts=5,
        )
        output_1d = mo.vstack([_controls, _fig_1d])
    else:
        output_1d = mo.vstack([_controls, mo.md("*No fields selected or no data loaded.*")])

    output_1d
    return


@app.cell
def _(mo):
    mo.md("## Integrated Moments (Saturation Check)")

    moment_selector = mo.ui.dropdown(
        options={
            'Electron Particles': 'int_n_elc',
            'Ion Particles':      'int_n_ion',
            'Electron Energy':    'int_en_elc',
            'Ion Energy':         'int_en_ion'
        },
        # Change 'int_en_elc' to the matching Key/Label:
        value='Electron Energy',
        label="Select Integrated Quantity"
    )

    norm_toggle = mo.ui.checkbox(value=True, label="Normalize to initial value")

    mo.hstack([moment_selector, norm_toggle])
    return moment_selector, norm_toggle


@app.cell(hide_code=True)
def _(comparison_data, moment_selector, norm_toggle, plt):
    if not comparison_data:
        output_int = "No data loaded."
    else:
        fig, ax = plt.subplots(figsize=(10, 5))

        # Helper to get the label for the title
        pretty_name = next(k for k, v in moment_selector.options.items() if v == moment_selector.value)

        for label, d in comparison_data.items():
            res = d['results']

            # --- FIXED LINE BELOW ---
            time = res.get('time_series_t')
            # ------------------------

            data = res.get(moment_selector.value)

            if time is not None and data is not None:
                # Convert time to microseconds
                t_plot = time * 1e6

                # Normalize if checkbox is checked
                y_plot = data / data[0] if norm_toggle.value else data

                ax.plot(
                    t_plot,
                    y_plot,
                    label=label,
                    color=d.get('color'),
                    linestyle=d.get('ls')
                )

        ax.set_xlabel("Time ($\mu s$)")
        ax.set_ylabel("Relative Value" if norm_toggle.value else "Value")
        ax.set_title(f"Comparison: {pretty_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        output_int = fig

    output_int
    return


@app.cell
def _(fend_input, fstart_input, mo):
    mo.md("## 🖼️ 2D Turbulence Snapshots")

    # 2D Controls
    frame_slider = mo.ui.slider(
        start=fstart_input.value,
        stop=fend_input.value,
        value=fend_input.value,
        step=1,
        label="Frame Index"
    )

    field_2d_select = mo.ui.dropdown(
        options=['ne', 'Te', 'phi', 'ni', 'Ti'],
        value='ne',
        label="Field"
    )

    mode_toggle = mo.ui.radio(
        options=['total', 'fluctuation'],
        value='fluctuation',
        label="Mode"
    )

    mo.hstack([frame_slider, field_2d_select, mode_toggle], justify="start")
    return field_2d_select, frame_slider, mode_toggle


@app.cell
def _(
    amu_input,
    field_2d_select,
    frame_slider,
    lcfs_input,
    mlib,
    mode_toggle,
    sim_definitions,
):
    # 2. Setup args
    vlims = [-0.5,0.5] if mode_toggle.value == 'fluctuation' else None

    # 3. Generate Figure using 'mlib'
    fig_2d = mlib.plot_2d_comparison(
        sim_definitions,
        frame=frame_slider.value,
        field_name=field_2d_select.value,
        mode=mode_toggle.value,
        lcfs_shift=lcfs_input.value,
        vlims=vlims,
        amu=amu_input.value
    )

    # 4. FINAL LINE: Return the figure object
    fig_2d
    return


@app.cell
def _(mo):
    mo.md("## SOL Heat Flux Decay ($\lambda_q$)")

    fit_rmin_input = mo.ui.number(value=0.00, step=0.005, label="Fit Start (R - LCFS) [m]")
    fit_rmax_input = mo.ui.number(value=0.05, step=0.005, label="Fit End (R - LCFS) [m]")
    return fit_rmax_input, fit_rmin_input


@app.cell
def _(
    comparison_data,
    fit_rmax_input,
    fit_rmin_input,
    lcfs_input,
    mlib,
    mo,
    trim_input,
):
    _sol_controls = mo.hstack([fit_rmin_input, fit_rmax_input])

    if comparison_data:
        _fig_sol = mlib.plot_qpara_sol(
            comparison_data, 
            lcfs_shift=lcfs_input.value,
            x_trim=trim_input.value*2,
            fit_rmin=fit_rmin_input.value,
            fit_rmax=fit_rmax_input.value
        )
        sol_output = mo.vstack([_sol_controls, _fig_sol])
    else:
        sol_output = mo.vstack([_sol_controls, mo.md("*No data loaded.*")])

    sol_output
    return


@app.cell
def _(comparison_data, lcfs_input, mo, np):
    mo.md("## Parallel Mode Structure & Local Shear")

    # Use protected local variables (_) to avoid Marimo conflicts
    _min_r, _max_r, _default_r = 0.0, 1.0, 0.0

    if comparison_data:
        _first_lbl = list(comparison_data.keys())[0]
        if 'x' in comparison_data[_first_lbl]:
            # Calculate coordinate as R - LCFS
            _x_arr = comparison_data[_first_lbl]['x'] - lcfs_input.value
            _min_r = float(np.round(_x_arr.min(), 3))
            _max_r = float(np.round(_x_arr.max(), 3))
            # Default to the separatrix (0.0) if it's within our domain
            _default_r = float(np.clip(0.0, _min_r, _max_r)) 

    # Create the slider
    z_r_slider = mo.ui.slider(
        start=_min_r, stop=_max_r, value=_default_r, step=0.01, 
        label="Radial Location (R - R_LCFS) [m]"
    )
    return (z_r_slider,)


@app.cell(hide_code=True)
def _(comparison_data, lcfs_input, mlib, mo, z_r_slider):
    # Render the slider and the plot together
    if comparison_data:
        _fig_parallel = mlib.plot_parallel_mode_structure(
            comparison_data,
            lcfs_shift=lcfs_input.value,
            r_target_val=z_r_slider.value
        )

        # Display the slider right above the figure
        output_parallel = mo.vstack([
            z_r_slider,
            _fig_parallel
        ])
    else:
        output_parallel = mo.md("*No data loaded.*")

    output_parallel
    return


@app.cell
def _(comparison_data, mo):
    mo.md("## Spectral Analysis ($k_y$)")

    # Get max radial index for the slider
    _max_nx = 0
    if comparison_data:
        _first_sim = list(comparison_data.keys())[0]
        if 'x' in comparison_data[_first_sim]:
            _max_nx = len(comparison_data[_first_sim]['x']) - 1

    spectra_x_idx = mo.ui.slider(
        start=0, stop=_max_nx, value=_max_nx//2, step=1, 
        label="Select Radial Index (x_idx) for spectra"
    )

    spectra_plot_type = mo.ui.dropdown(
        options=['Amplitudes', 'Fluxes & Phases'],
        value='Amplitudes',
        label="Plot View"
    )
    return spectra_plot_type, spectra_x_idx


@app.cell
def _(
    amu_input,
    fend_input,
    fstart_input,
    mlib,
    mo,
    sim_definitions,
    spectra_x_idx,
):
    @mo.cache
    def load_spectra_batch(sims, f_start, f_end, x_id, mass):
        _output = {}
        for _label, _meta in sims.items():
            _path = _meta.get('path') if isinstance(_meta, dict) else _meta
            _res = mlib.process_spectra_for_sim(_path, f_start, f_end, x_id, step=10, amu=mass)
            if _res is not None:
                _output[_label] = _res
        return _output

    with mo.status.spinner("Computing FSA Spectra (this may take a moment)..."):
        spectra_data = load_spectra_batch(
            sim_definitions, 
            fstart_input.value, 
            fend_input.value, 
            spectra_x_idx.value, 
            amu_input.value,
        )
    return (spectra_data,)


@app.cell
def _(mlib, mo, spectra_data, spectra_plot_type, spectra_x_idx):
    mo.vstack([
        mo.hstack([spectra_x_idx, spectra_plot_type]),
        mlib.plot_spectra_dashboard(spectra_data, plot_mode=spectra_plot_type.value) if spectra_data else mo.md("*No spectra data available.*")
    ])
    return


@app.cell
def _(comparison_data, mo):
    mo.md("## 📊 PDF of Fluctuations at a Radial Slice")

    # Determine max_nx for the slider based on loaded data
    _max_nx = 0
    if comparison_data:
        _first_sim = list(comparison_data.keys())[0]
        if 'x' in comparison_data[_first_sim]:
            _max_nx = len(comparison_data[_first_sim]['x']) - 1

    # Explicitly map the options to bypass FIELD_INFO completely
    pdf_field_selector = mo.ui.dropdown(
        options={
            "Density": "dn_flat",
            "Electron Temp": "dT_flat",
            "Potential": "dphi_flat"
        },
        value='Density',
        label="Select PDF Fluctuation Field"
    )

    pdf_x_idx_selector = mo.ui.slider(
        0, _max_nx, value=_max_nx // 2, label="Radial Index (x_idx)"
    )

    pdf_gaussian_toggle = mo.ui.checkbox(
        value=True,
        label="Compare to Gaussian PDF"
    )
    return pdf_field_selector, pdf_gaussian_toggle, pdf_x_idx_selector


@app.cell
def _(
    comparison_data,
    lcfs_input,
    mlib,
    mo,
    pdf_field_selector,
    pdf_gaussian_toggle,
    pdf_x_idx_selector,
):
    _pdf_controls = mo.vstack([
        pdf_field_selector,
        pdf_x_idx_selector,
        pdf_gaussian_toggle
    ])

    # Generate the plot
    _output_plot = None
    if comparison_data:
        _first_lbl = list(comparison_data.keys())[0]
        if pdf_field_selector.value in comparison_data[_first_lbl]['results']:
            _output_plot = mlib.plot_pdf_slice(
                comparison_data,
                field_key=pdf_field_selector.value,
                bins='auto',
                density=True,
                compare_gaussian=pdf_gaussian_toggle.value,
                lcfs_shift=lcfs_input.value,
                x_target_idx=pdf_x_idx_selector.value
            )
        else:
            _output_plot = mo.md(f"*Selected PDF field '{pdf_field_selector.value}' not available in results.*")
    else:
        _output_plot = mo.md("*No data loaded.*")

    # Output both the controls (slider) and the plot to the screen
    pdf_final_output = mo.vstack([_pdf_controls, _output_plot])

    pdf_final_output
    return


@app.cell
def _(comparison_data, lcfs_input, mo, np):
    mo.md("## ⏱️ Frequency Spectra ($\omega$)")

    # Calculate Slider Bounds based on R - LCFS
    _min_r, _max_r, _default_r = 0.0, 1.0, 0.0
    if comparison_data:
        _first_lbl = list(comparison_data.keys())[0]
        if 'x' in comparison_data[_first_lbl]:
            _x_arr = comparison_data[_first_lbl]['x'] - lcfs_input.value
            _min_r = float(np.round(_x_arr.min(), 3))
            _max_r = float(np.round(_x_arr.max(), 3))
            _default_r = float(np.clip(0.0, _min_r, _max_r))

    freq_r_slider = mo.ui.slider(
        start=_min_r, stop=_max_r, value=_default_r, step=0.01, 
        label="Radial Location (R - R_LCFS) [m]"
    )

    freq_field_select = mo.ui.dropdown(
        options={"Density": "ne", "Electron Temp": "Te"},
        value="Density",
        label="Select Cross-Spectrum"
    )
    return freq_field_select, freq_r_slider


@app.cell
def _(comparison_data, freq_field_select, freq_r_slider, lcfs_input, mlib, mo):
    # Stack the UI
    _freq_controls = mo.vstack([freq_field_select, freq_r_slider])

    # Generate the plot
    if comparison_data:
        _fig_freq = mlib.plot_frequency_spectra(
            comparison_data,
            field_key=freq_field_select.value,
            r_target_val=freq_r_slider.value,
            lcfs_shift=lcfs_input.value
        )
        _plot_output = _fig_freq
    else:
        _plot_output = mo.md("*No data loaded.*")

    # Combine and display!
    freq_final_output = mo.vstack([_freq_controls, _plot_output])
    freq_final_output
    return


if __name__ == "__main__":
    app.run()
