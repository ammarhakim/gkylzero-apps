import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

def load_spectrum(h5file, quantity):
    with h5py.File(h5file, 'r') as f:
        ky = f['ky_binormal'][:]
        if quantity not in f:
            raise ValueError(f"Quantity '{quantity}' not found in {h5file}")
        spectrum = f[quantity][:]
    return ky, spectrum

def main():
    parser = argparse.ArgumentParser(
        description="Compare k_y spectra from multiple simulations."
    )
    parser.add_argument('files', nargs='+', help='HDF5 spectra files to compare')
    parser.add_argument('--quantity', type=str, default='phi', help='Quantity to compare (e.g., phi, elcDens, ionDens, elcTemp, ionTemp)')
    parser.add_argument('--labels', nargs='*', help='Labels for each simulation (default: filenames)')
    parser.add_argument('--save', type=str, default=None, help='Filename to save the plot (optional)')
    args = parser.parse_args()

    plt.figure(figsize=(8,6))
    for idx, h5file in enumerate(args.files):
        label = args.labels[idx] if args.labels and idx < len(args.labels) else h5file
        ky, spectrum = load_spectrum(h5file, args.quantity)
        ky_shifted = np.fft.fftshift(ky)
        spectrum_shifted = np.fft.fftshift(spectrum)
        mask = ky_shifted > 0
        norm = np.max(spectrum_shifted[mask])
        plt.plot(ky_shifted[mask], spectrum_shifted[mask]/norm, label=label)

    plt.xlabel(r'$k_y$ (Binormal Wavenumber)')
    plt.ylabel('Normalized Power Spectrum')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.title(f"Comparison of $k_y$ Spectra ({args.quantity})")
    plt.tight_layout()
    if args.save:
        plt.savefig(args.save)
        print(f"Saved plot to {args.save}")
    plt.show()

if __name__ == "__main__":
    main()