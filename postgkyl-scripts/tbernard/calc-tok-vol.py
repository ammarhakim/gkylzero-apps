import math

def miller_plasma_volume(R, a, kappa, delta, sol_thickness=0.0):
    """
    Estimate tokamak plasma volume using Miller geometry.

    Parameters:
        R (float): Major radius (m)
        a (float): Minor radius (m)
        kappa (float): Elongation
        delta (float): Triangularity (0 <= delta < 1)
        sol_thickness (float): Thickness of scrape-off layer (m)

    Returns:
        dict: Volumes in cubic meters
            - core_volume
            - total_volume (core + SOL)
            - sol_volume (just the SOL region)
    """
    f_delta = 1 - 0.5 * delta**2

    # Core plasma volume
    core_volume = 2 * math.pi**2 * R * a**2 * kappa * f_delta

    # Effective minor radius with SOL
    a_eff = a + sol_thickness
    total_volume = 2 * math.pi**2 * R * a_eff**2 * kappa * f_delta

    # SOL volume is the difference
    sol_volume = total_volume - core_volume

    return {
        "core_volume_m3": core_volume,
        "total_volume_m3": total_volume,
        "sol_volume_m3": sol_volume
    }

# Example use
if __name__ == "__main__":
    R = 1.7         # Major radius (m)
    a = 0.6         # Minor radius (m)
    kappa = 1.35     # Elongation
    delta = -0.4     # Triangularity
    sol = 0.05      # SOL thickness (m)

    volumes = miller_plasma_volume(R, a, kappa, delta, sol)

    print("Core Plasma Volume: {:.2f} m³".format(volumes["core_volume_m3"]))
    print("Total Plasma + SOL Volume: {:.2f} m³".format(volumes["total_volume_m3"]))
    print("SOL Volume: {:.2f} m³".format(volumes["sol_volume_m3"]))
