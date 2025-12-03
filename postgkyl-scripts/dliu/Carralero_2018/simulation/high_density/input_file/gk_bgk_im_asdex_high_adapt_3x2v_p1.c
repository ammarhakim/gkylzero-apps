#include <math.h>
#include <stdio.h>
#include <time.h>

#include <gkyl_alloc.h>
#include <gkyl_const.h>
#include <gkyl_eqn_type.h>
#include <gkyl_fem_poisson_bctype.h>
#include <gkyl_gyrokinetic.h>
#include <gkyl_gyrokinetic_run.h>
#include <gkyl_math.h>

#include <rt_arg_parse.h>

// Define the context of the simulation. This stores global parameters.
struct gk_app_ctx {
  int cdim, vdim;
  // Plasma parameters
  int num_species;
  double me, qe, mi, qi;
  // Initial conditions.
  double den_upstream_min; 
  double den_upstream_max; 
  double den_floor       ; 
  double sig_rho_den     ; 

  double Te_upstream_min ; 
  double Te_upstream_max ; 
  double Te_floor        ; 
  double sig_rho_Te      ; 

  double Ti_upstream_min ; 
  double Ti_upstream_max ; 
  double Ti_floor        ; 
  double sig_rho_Ti      ; 
  double n0, Te0, Ti0, B0; // Reference parameters.
  // Collision parameters
  double nuFrac, nuElc, nuIon;
  // Source parameters
  int num_sources;
  bool adapt_energy_srcCORE, adapt_particle_srcCORE;
  double center_srcCORE[3], sigma_srcCORE[3];
  double energy_srcCORE, particle_srcCORE;
  double floor_srcCORE;
  bool adapt_energy_srcWALL, adapt_particle_srcWALL;
  double center_srcWALL[3], sigma_srcWALL[3];
  double energy_srcWALL, particle_srcWALL;
  double floor_srcWALL;
  // Grid parameters
  char eqdsk_file[128];
  double psi_axis;
  double psi_sep;
  double Lx, Ly, Lz;
  double x_min, x_max, y_min, y_max, z_min, z_max, rho_min, rho_max;
  int num_cell_x, num_cell_y, num_cell_z, num_cell_vpar, num_cell_mu;
  int cells[GKYL_MAX_DIM], poly_order;
  double vpar_max_elc, mu_max_elc, vpar_max_ion, mu_max_ion;
  // Simulation control parameters
  double final_time, write_phase_freq;
  int num_frames, int_diag_calc_num, num_failures_max;
  double dt_failure_tol;
};

double rho_psi(double psi, double psi_axis, double psi_sep)
{
  // Normalized radial coordinate.
  return sqrt((psi-psi_axis) / (psi_sep - psi_axis));
}

double psi_rho(double rho, double psi_axis, double psi_sep)
{
  // Poloidal flux given the normalized radial coordinate.
  return pow(rho,2) * (psi_sep - psi_axis) + psi_axis;
}

// Density initial condition (like AUG exp profile)
void eval_density(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  double x = xn[0], z = xn[2];

  struct gk_app_ctx *app = ctx;
  double psi_axis = app->psi_axis;
  double psi_sep = app->psi_sep;
  double rho_min = app->rho_min;
  double sig_rho_den = app->sig_rho_den;
  double den_upstream_max = app->den_upstream_max;
  double den_floor = app->den_floor;

  double rho = rho_psi(x, psi_axis, psi_sep);

  // Use a constant in z profile because in reality this is a semi-detached
  // case for which the density actually increases towards the divertor plates,
  // but we don't have the physics for that. The simulation will likely produce
  // a decreasing density profile towards the plates.
  double profile = den_upstream_max * exp(-(rho - rho_min)/sig_rho_den);

  fout[0] = fmax(profile, den_floor);
}

// Flow initial condition
void
eval_upar(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  fout[0] = 0.0;
}

// Electron temperature initial conditions
void eval_temp_elc(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  double x = xn[0], z = xn[2];

  struct gk_app_ctx *app = ctx;
  double psi_axis = app->psi_axis;
  double psi_sep = app->psi_sep;
  double rho_min = app->rho_min;
  double sig_rho_Te = app->sig_rho_Te;
  double Te_upstream_max = app->Te_upstream_max;
  double Te_floor = app->Te_floor;

  double rho = rho_psi(x, psi_axis, psi_sep);

  double profile = Te_upstream_max * exp(-(rho - rho_min)/sig_rho_Te);

  // Multiply by a function that smoothly drops the temperature by 4X towards
  // the divertor plates.
  profile *= 0.25+0.75*exp(-0.5*pow(z/(M_PI/1.35),8));

  fout[0] = fmax(profile, Te_floor);
}

// Ion temperature initial conditions
void eval_temp_ion(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  double x = xn[0], z = xn[2];

  struct gk_app_ctx *app = ctx;
  double psi_axis = app->psi_axis;
  double psi_sep = app->psi_sep;
  double rho_min = app->rho_min;
  double sig_rho_Ti = app->sig_rho_Ti;
  double Ti_upstream_max = app->Ti_upstream_max;
  double Ti_floor = app->Ti_floor;

  double rho = rho_psi(x, psi_axis, psi_sep);

  double profile = Ti_upstream_max * exp(-(rho - rho_min)/sig_rho_Ti);

  // Multiply by a function that smoothly drops the temperature by 4X towards
  // the divertor plates.
  profile *= 0.25+0.75*exp(-0.5*pow(z/(M_PI/1.35),8));

  fout[0] = fmax(profile, Ti_floor);
}

// Taken from rt gk d3d 3x2c, is this the non uniform v grid mapping?
void mapc2p_vel_elc(double t, const double *vc, double* GKYL_RESTRICT vp, void *ctx)
{
  struct gk_app_ctx *app = ctx;
  double vpar_max_elc = app->vpar_max_elc;
  double mu_max_elc = app->mu_max_elc;
  double cvpar = vc[0], cmu = vc[1];
  // Linear map up to vpar_max/2, then quadratic.
  if (fabs(cvpar) <= 0.5)
    vp[0] = vpar_max_elc*cvpar;
  else if (cvpar < -0.5)
    vp[0] = -vpar_max_elc*2.0*pow(cvpar,2);
  else
    vp[0] =  vpar_max_elc*2.0*pow(cvpar,2);
  // Quadratic map in mu.
  vp[1] = mu_max_elc*pow(cmu,2);
}

void mapc2p_vel_ion(double t, const double *vc, double* GKYL_RESTRICT vp, void *ctx)
{
  struct gk_app_ctx *app = ctx;
  double vpar_max_ion = app->vpar_max_ion;
  double mu_max_ion = app->mu_max_ion;
  double cvpar = vc[0], cmu = vc[1];
  // Linear map up to vpar_max/2, then quadratic.
  if (fabs(cvpar) <= 0.5)
    vp[0] = vpar_max_ion*cvpar;
  else if (cvpar < -0.5)
    vp[0] = -vpar_max_ion*2.0*pow(cvpar,2);
  else
    vp[0] =  vpar_max_ion*2.0*pow(cvpar,2);
  // Quadratic map in mu.
  vp[1] = mu_max_ion*pow(cmu,2);
}

struct gk_app_ctx create_ctx(void)
{
  int cdim = 3, vdim = 2; // Dimensionality
  // Universal constant parameters.
  double eps0 = GKYL_EPSILON0, eV = GKYL_ELEMENTARY_CHARGE;
  double proton_mass = GKYL_PROTON_MASS, electron_mass = GKYL_ELECTRON_MASS;

  // Location of the numerical equilibrium.
  char eqdsk_file[128] = "../../../experiment/33341/Equilibria/High_density/33341_3.592.eqdsk";

  // Get info from eqdsk file.
  struct gkyl_efit_inp efit_inp = {
    .rz_poly_order = 2,
    .flux_poly_order = 1,
  };
  memcpy(efit_inp.filepath, eqdsk_file, sizeof(eqdsk_file));
  struct gkyl_efit *efit = gkyl_efit_new(&efit_inp);
  double R_axis = efit->rmaxis; // R of the magnetic axis.
  double Z_axis = efit->zmaxis; // Z of the magnetic axis.
  double psi_sep = efit->psisep; // psi at the separatrix.
  double psi_axis = efit->simag; // psi at the magnetic axis.
  double Rxpt = efit->Rxpt[0], Zxpt = efit->Zxpt[0];
  gkyl_efit_release(efit);

  double bmag_min = 1.935764e+00;
  double bmag_max = 3.913808e+00;

  // The radial extents
  //  x_min = 0.181
  //  x_max = 0.201
  // give 
  //  R_omp_min = 2.14397
  //  R_omp_max = 2.17442
  // so a 3 cm radial box at the OMP.
  double x_min = 0.181; // Minimum psi of the simulation domain.
  double x_max = 0.201; // Maximum psi of the simulation domain.
  double R_omp_min = 2.14397; // Min major radius at the OMP.
  double R_omp_max = 2.17442; // Max major radius at the OMP.

  double z_min = -(M_PI-1e10);
  double z_max =   M_PI-1e10;

  double rho_min = rho_psi(x_min, psi_axis, psi_sep);
  double rho_max = rho_psi(x_max, psi_axis, psi_sep);

  // Species mass and charge.
  int num_species = 2;
  double me = electron_mass;
  double mi = proton_mass*2.01410177811; // Deuterium ions
  double qi =  eV; // ion charge
  double qe = -eV; // electron charge

  // Parameters controlling initial conditions.
  double den_upstream_min = 1.00e19; // [1/m^3]
  double den_upstream_max = 1.44e19; // [1/m^3]
  double den_floor = 0.05*den_upstream_max; // Min density in IC.
  double sig_rho_den = 2.4*(rho_max - rho_min); // Exp delay length in density.

  double Te_upstream_min = 16.0*eV; // [J]
  double Te_upstream_max = 30.0*eV; // [J]
  double Te_floor = 0.05*Te_upstream_max; // Min Te in IC.
  double sig_rho_Te = 1.2*(rho_max - rho_min); // Exp delay length in Te.

  double Ti_upstream_min = 20.0*eV; // [J]
  double Ti_upstream_max = 125.0*eV; // [J]
  double Ti_floor = 0.05*Ti_upstream_max; // Min Ti in IC.
  double sig_rho_Ti = 0.5*(rho_max - rho_min); // Exp delay length in Ti.

  // Reference parameters.
  double n0  = 0.5*(den_upstream_min+den_upstream_max);
  double Te0 = 0.5*(Te_upstream_min+Te_upstream_max);
  double Ti0 = 0.5*(Ti_upstream_min+Ti_upstream_max);
  double B0  = 0.5*(bmag_min+bmag_max);

  double vte = sqrt(Te0/me);
  double vti = sqrt(Ti0/mi);
  double c_s = sqrt(Te0/mi);
  double omega_ci = fabs(qi*B0/mi);
  double rho_s = c_s/omega_ci;

  double q_min = 3.862429e+00; 
  double q_max = 4.768886e+00; 
  double q0 = 0.5*(q_min+q_max);
  double r0 = 0.5*(R_omp_min+R_omp_max)-R_axis;
  double Ly = 100*rho_s*q0/r0;
  double y_min = -Ly/2.;
  double y_max =  Ly/2.;

  double Lx = x_max - x_min;
  double Lz = z_max - z_min;

  // The limits
  // Source parameters
  int num_sources = 2;
  double P_exp = 0.47e6; // P_sol measured [W]
  double vol_frac = Ly/(2.0*M_PI); // Volume fraction of the simulation box
  double P_inj = P_exp * vol_frac / num_species;
  // Core source:
  // - The particles injection is only the one that are lost through the inner radial boundary.
  bool adapt_energy_srcCORE = true; // The source will compensate the losses in energy according to given boundaries.
  bool adapt_particle_srcCORE = true; // The source will compensate the losses in particle according to given boundaries.
  double energy_srcCORE = 0.0; // What the source must inject in energy [W]
  double particle_srcCORE = 0.0;// What the source must inject in particle [1/s]
  double center_srcCORE[3] = {x_min, 0.0, -Lz/4}; // This is the position of the ion source,
  double sigma_srcCORE[3] = {0.06*Lx, 0.0, Lz/8}; //  the electron source will be at +Lz/2.
  double floor_srcCORE = 1e-10;
  // Wall source:
  // - Injects energy only in the wall region.
  // - Reinjects particles that are absorbed by the wall.
  // - Energy is free to leave the system.
  bool adapt_energy_srcWALL = false; // The source will compensate the losses in energy according to given boundaries.
  bool adapt_particle_srcWALL = true; // The source will compensate the losses in particle according to given boundaries.
  double energy_srcWALL = P_inj; // What the source must inject in energy [W]
  double particle_srcWALL = 0.0;// What the source must inject in particle [1/s]
  double center_srcWALL[3] = {x_min, 0.0, -Lz/4}; // This is the position of the ion source,
  double sigma_srcWALL[3] = {0.06*Lx, 0.0, Lz/8}; //  the electron source will be at +Lz/2.
  double floor_srcWALL = 1e-10;

  // Grid parameters
  int num_cell_x = 32; 
  int num_cell_y = 32;
  int num_cell_z = 24;
  int num_cell_vpar = 16;
  int num_cell_mu = 8;
  int poly_order = 1;

  // Velocity box dimensions
  double vpar_max_elc = 6.*vte;
  double mu_max_elc   = me*pow(4*vte,2)/(2*B0);
  double vpar_max_ion = 6.*vti;
  double mu_max_ion   = mi*pow(4*vti,2)/(2*B0);

  double final_time = 2.e-3;
  int num_frames = 2000;
  double write_phase_freq = 0.01;
  int int_diag_calc_num = num_frames*100;
  double dt_failure_tol = 1.0e-3; // Minimum allowable fraction of initial time-step.
  int num_failures_max = 20; // Maximum allowable number of consecutive small time-steps.

  struct gk_app_ctx ctx = {
    .cdim = cdim,
    .vdim = vdim,
    .psi_sep  = psi_sep ,
    .psi_axis = psi_axis,
    .x_min = x_min,  .x_max = x_max,
    .y_min = y_min,  .y_max = y_max,
    .z_min = z_min,  .z_max = z_max,
    .Lx = Lx,
    .Ly = Ly,
    .Lz = Lz,
    .rho_min = rho_min,  .rho_max = rho_max,
    .num_species = num_species,
    .me = me,  .qe = qe,
    .mi = mi,  .qi = qi,
    .den_upstream_min = den_upstream_min,
    .den_upstream_max = den_upstream_max,
    .den_floor        = den_floor       ,
    .sig_rho_den      = sig_rho_den     ,
    .Te_upstream_min  = Te_upstream_min ,
    .Te_upstream_max  = Te_upstream_max ,
    .Te_floor         = Te_floor        ,
    .sig_rho_Te       = sig_rho_Te      ,
    .Ti_upstream_min  = Ti_upstream_min ,
    .Ti_upstream_max  = Ti_upstream_max ,
    .Ti_floor         = Ti_floor        ,
    .sig_rho_Ti       = sig_rho_Ti      ,
    .n0 = n0,  .Te0 = Te0,  .Ti0 = Ti0,
    .B0 = B0,
    .num_sources = num_sources,
    .adapt_energy_srcCORE = adapt_energy_srcCORE,
    .adapt_particle_srcCORE = adapt_particle_srcCORE,
    .center_srcCORE = {center_srcCORE[0], center_srcCORE[1], center_srcCORE[2]},
    .sigma_srcCORE = {sigma_srcCORE[0], sigma_srcCORE[1], sigma_srcCORE[2]},
    .energy_srcCORE = energy_srcCORE,  .particle_srcCORE = particle_srcCORE,
    .floor_srcCORE = floor_srcCORE,
    .adapt_energy_srcWALL = adapt_energy_srcWALL,
    .adapt_particle_srcWALL = adapt_particle_srcWALL,
    .center_srcWALL = {center_srcWALL[0], center_srcWALL[1], center_srcWALL[2]},
    .sigma_srcWALL = {sigma_srcWALL[0], sigma_srcWALL[1], sigma_srcWALL[2]},
    .energy_srcWALL = energy_srcWALL,  .particle_srcWALL = particle_srcWALL,
    .floor_srcWALL = floor_srcWALL,
    .num_cell_x     = num_cell_x,
    .num_cell_y     = num_cell_y,
    .num_cell_z     = num_cell_z,
    .num_cell_vpar  = num_cell_vpar,
    .num_cell_mu    = num_cell_mu,
    .cells = {num_cell_x, num_cell_y, num_cell_z, num_cell_vpar, num_cell_mu},
    .poly_order   = poly_order,
    .vpar_max_elc = vpar_max_elc,  .mu_max_elc = mu_max_elc,
    .vpar_max_ion = vpar_max_ion,  .mu_max_ion = mu_max_ion,
    .write_phase_freq = write_phase_freq,
    .final_time = final_time,  .num_frames = num_frames,
    .int_diag_calc_num = int_diag_calc_num,
    .dt_failure_tol = dt_failure_tol,
    .num_failures_max = num_failures_max,
  };

  // Copy eqdsk file into ctx.
  memcpy(ctx.eqdsk_file, eqdsk_file, sizeof(eqdsk_file));

  return ctx;
}

int 
main(int argc, char **argv)
{
  struct gkyl_app_args app_args = parse_app_args(argc, argv);

#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi) MPI_Init(&argc, &argv);
#endif
  
  if (app_args.trace_mem) {
    gkyl_cu_dev_mem_debug_set(true);
    gkyl_mem_debug_set(true);
  }

  struct gk_app_ctx ctx = create_ctx(); // context for init functions

  int cells_x[ctx.cdim], cells_v[ctx.vdim];
  for (int d=0; d<ctx.cdim; d++)
    cells_x[d] = APP_ARGS_CHOOSE(app_args.xcells[d], ctx.cells[d]);
  for (int d=0; d<ctx.vdim; d++)
    cells_v[d] = APP_ARGS_CHOOSE(app_args.vcells[d], ctx.cells[ctx.cdim+d]);

  // Construct communicator for use in app.
  struct gkyl_comm *comm = gkyl_gyrokinetic_comms_new(app_args.use_mpi, app_args.use_gpu, stderr);

  // Electron core source:
  struct gkyl_gyrokinetic_projection proj_srcCORE_e = {
    .proj_id = GKYL_PROJ_MAXWELLIAN_GAUSSIAN,
    .gaussian_mean = {ctx.center_srcCORE[0], ctx.center_srcCORE[1], ctx.center_srcCORE[2]},
    .gaussian_std_dev = {ctx.sigma_srcCORE[0], ctx.sigma_srcCORE[1], ctx.sigma_srcCORE[2]},
    .total_num_particles = ctx.particle_srcCORE,
    .total_kin_energy = ctx.energy_srcCORE,
    .temp_max = 5.0*ctx.Te0,
    .temp_min = 0.1*ctx.Te0,
    .f_floor = ctx.floor_srcCORE,
  };
  struct gkyl_gyrokinetic_adapt_source adapt_srcCORE_e ={
    .adapt_to_species = "elc",
    .adapt_particle = ctx.adapt_particle_srcCORE,
    .adapt_energy = ctx.adapt_energy_srcCORE,
    .num_boundaries = 1,
    .dir = {0},
    .edge = {GKYL_LOWER_EDGE},
  };
  // Electron wall source:
  struct gkyl_gyrokinetic_projection proj_srcWALL_e = {
    .proj_id = GKYL_PROJ_MAXWELLIAN_GAUSSIAN,
    .gaussian_mean = {ctx.center_srcWALL[0], ctx.center_srcWALL[1], ctx.center_srcWALL[2]},
    .gaussian_std_dev = {ctx.sigma_srcWALL[0], ctx.sigma_srcWALL[1], ctx.sigma_srcWALL[2]},
    .total_num_particles = ctx.particle_srcWALL,
    .total_kin_energy = ctx.energy_srcWALL,
    .temp_max = 5.0*ctx.Te0,
    .temp_min = 0.1*ctx.Te0,
    .f_floor = ctx.floor_srcWALL,
  };
  struct gkyl_gyrokinetic_adapt_source adapt_srcWALL_e ={
    .adapt_to_species = "elc",
    .adapt_particle = ctx.adapt_particle_srcWALL,
    .adapt_energy = ctx.adapt_energy_srcWALL,
    .num_boundaries = 3,
    .dir = {0, 2, 2},
    .edge = {GKYL_UPPER_EDGE, GKYL_LOWER_EDGE, GKYL_UPPER_EDGE},
  };

  // electrons
  struct gkyl_gyrokinetic_species elc = {
    .name = "elc",
    .charge = ctx.qe, .mass = ctx.me,
    .vdim = ctx.vdim,
    .lower = { -1.0/sqrt(2.0), 0.0},
    .upper = {  1.0/sqrt(2.0), 1.0},
    .cells = { cells_v[0], cells_v[1] },
    .polarization_density = ctx.n0,

    .mapc2p = {
      .mapping = mapc2p_vel_elc,
      .ctx = &ctx,
    },

    .projection = {
      .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
      .ctx_density = &ctx,
      .ctx_upar = &ctx,
      .ctx_temp = &ctx,
      .density = eval_density,
      .upar = eval_upar,
      .temp = eval_temp_elc,
    },

//    .positivity = {
//      .type = GKYL_GK_POSITIVITY_SHIFT,
//      .quasineutrality_rescale = true,
//      .write_diagnostics = true,
//    },

    .collisionless = {
      .type = GKYL_GK_COLLISIONLESS_ES,
    },

    .correct = {
      .correct_all_moms = true,
      .use_last_converged = true,
      .iter_eps = 1e-12,
      .max_iter = 10,
    },

    .collisions =  {
      .collision_id = GKYL_BGK_COLLISIONS,
      .den_ref = ctx.n0, // Density used to calculate coulomb logarithm
      .temp_ref = ctx.Te0, // Temperature used to calculate coulomb logarithm
      .num_cross_collisions = 1,
      .collide_with = { "ion" },
      .is_implicit = true,
    },

    .source = {
      .source_id = GKYL_PROJ_SOURCE,
      .num_sources = ctx.num_sources,
      .num_adapt_sources = ctx.num_sources,
      .projection[0] = proj_srcCORE_e,
      .adapt[0] = adapt_srcCORE_e,
      .projection[1] = proj_srcWALL_e,
      .adapt[1] = adapt_srcWALL_e,
      .diagnostics = {
        .num_diag_moments = 1,
        .diag_moments = {GKYL_F_MOMENT_HAMILTONIAN},
        .num_integrated_diag_moments = 1,
        .integrated_diag_moments = {GKYL_F_MOMENT_HAMILTONIAN},
      }
    },

    .bcs = {
      { .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB, },
      { .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB, },
      { .dir = 2, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_SHEATH, },
      { .dir = 2, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_SHEATH, },
    },

    .num_diag_moments = 10,
    .diag_moments = {GKYL_F_MOMENT_HAMILTONIAN, GKYL_F_MOMENT_MAXWELLIAN, GKYL_F_MOMENT_BIMAXWELLIAN,
      GKYL_F_MOMENT_M0, GKYL_F_MOMENT_M1, GKYL_F_MOMENT_M2PAR, GKYL_F_MOMENT_M2PERP, GKYL_F_MOMENT_M2,
      GKYL_F_MOMENT_M3PAR, GKYL_F_MOMENT_M3PERP},

    .num_integrated_diag_moments = 1,
    .integrated_diag_moments = { GKYL_F_MOMENT_HAMILTONIAN },

    .boundary_flux_diagnostics = {
      .num_diag_moments = 1,
      .diag_moments = {GKYL_F_MOMENT_HAMILTONIAN},
      .num_integrated_diag_moments = 1,
      .integrated_diag_moments = { GKYL_F_MOMENT_HAMILTONIAN },
    },

    .time_rate_diagnostics = true,
  };

  // Ion core source:
  struct gkyl_gyrokinetic_projection proj_srcCORE_i = {
    .proj_id = GKYL_PROJ_MAXWELLIAN_GAUSSIAN  ,
    .gaussian_mean = {ctx.center_srcCORE[0], ctx.center_srcCORE[1], ctx.center_srcCORE[2]},
    .gaussian_std_dev = {ctx.sigma_srcCORE[0], ctx.sigma_srcCORE[1], ctx.sigma_srcCORE[2]},
    .total_num_particles = ctx.particle_srcCORE,
    .total_kin_energy = ctx.energy_srcCORE,
    .temp_max = 5.0*ctx.Ti0,
    .temp_min = 0.1*ctx.Ti0,
    .f_floor = ctx.floor_srcCORE,
  };
  struct gkyl_gyrokinetic_adapt_source adapt_srcCORE_i ={
    .adapt_to_species = "ion",
    .adapt_particle = ctx.adapt_particle_srcCORE,
    .adapt_energy = ctx.adapt_energy_srcCORE,
    .num_boundaries = 1,
    .dir = {0},
    .edge = {GKYL_LOWER_EDGE},
  };
  // Ion wall source:
  struct gkyl_gyrokinetic_projection proj_srcWALL_i = {
    .proj_id = GKYL_PROJ_MAXWELLIAN_GAUSSIAN  ,
    .gaussian_mean = {ctx.center_srcWALL[0], ctx.center_srcWALL[1], ctx.center_srcWALL[2]},
    .gaussian_std_dev = {ctx.sigma_srcWALL[0], ctx.sigma_srcWALL[1], ctx.sigma_srcWALL[2]},
    .total_num_particles = ctx.particle_srcWALL,
    .total_kin_energy = ctx.energy_srcWALL,
    .temp_max = 5.0*ctx.Ti0,
    .temp_min = 0.1*ctx.Ti0,
    .f_floor = ctx.floor_srcWALL,
  };
  struct gkyl_gyrokinetic_adapt_source adapt_srcWALL_i ={
    .adapt_to_species = "ion",
    .adapt_particle = ctx.adapt_particle_srcWALL,
    .adapt_energy = ctx.adapt_energy_srcWALL,
    .num_boundaries = 3,
    .dir = {0, 2, 2},
    .edge = {GKYL_UPPER_EDGE, GKYL_LOWER_EDGE, GKYL_UPPER_EDGE},
  };

  // ions
  struct gkyl_gyrokinetic_species ion = {
    .name = "ion",
    .charge = ctx.qi, .mass = ctx.mi,
    .vdim = ctx.vdim,
    .lower = { -1.0/sqrt(2.0), 0.0},
    .upper = {  1.0/sqrt(2.0), 1.0},
    .cells = { cells_v[0], cells_v[1] },
    .polarization_density = ctx.n0,

    .mapc2p = {
      .mapping = mapc2p_vel_ion,
      .ctx = &ctx,
    },

    .projection = {
      .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
      .ctx_density = &ctx,
      .ctx_upar = &ctx,
      .ctx_temp = &ctx,
      .density = eval_density,
      .upar = eval_upar,
      .temp = eval_temp_ion,
    },

//    .positivity = {
//      .type = GKYL_GK_POSITIVITY_SHIFT,
//      .quasineutrality_rescale = true,
//      .write_diagnostics = true,
//    },

    .collisionless = {
      .type = GKYL_GK_COLLISIONLESS_ES,
    },

    .correct = {
      .correct_all_moms = true,
      .use_last_converged = true,
      .iter_eps = 1e-12,
      .max_iter = 10,
    },

    .collisions =  {
      .collision_id = GKYL_BGK_COLLISIONS,
      .den_ref = ctx.n0, // Density used to calculate coulomb logarithm
      .temp_ref = ctx.Ti0, // Temperature used to calculate coulomb logarithm
      .num_cross_collisions = 1,
      .collide_with = { "elc" },
      .is_implicit = true,
    },

    .source = {
      .source_id = GKYL_PROJ_SOURCE,
      .num_sources = ctx.num_sources,
      .num_adapt_sources = ctx.num_sources,
      .projection[0] = proj_srcCORE_i,
      .adapt[0] = adapt_srcCORE_i,
      .projection[1] = proj_srcWALL_e,
      .adapt[1] = adapt_srcWALL_e,
      .diagnostics = {
        .num_diag_moments = 1,
        .diag_moments = {GKYL_F_MOMENT_HAMILTONIAN},
        .num_integrated_diag_moments = 1,
        .integrated_diag_moments = {GKYL_F_MOMENT_HAMILTONIAN},
      }
    },

    .bcs = {
      { .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB, },
      { .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB, },
      { .dir = 2, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_SHEATH, },
      { .dir = 2, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_SHEATH, },
    },

    .num_diag_moments = 10,
    .diag_moments = {GKYL_F_MOMENT_HAMILTONIAN, GKYL_F_MOMENT_MAXWELLIAN, GKYL_F_MOMENT_BIMAXWELLIAN,
      GKYL_F_MOMENT_M0, GKYL_F_MOMENT_M1, GKYL_F_MOMENT_M2PAR, GKYL_F_MOMENT_M2PERP, GKYL_F_MOMENT_M2,
      GKYL_F_MOMENT_M3PAR, GKYL_F_MOMENT_M3PERP},
    
    .num_integrated_diag_moments = 1,
    .integrated_diag_moments = { GKYL_F_MOMENT_HAMILTONIAN },

    .boundary_flux_diagnostics = {
      .num_diag_moments = 1,
      .diag_moments = {GKYL_F_MOMENT_HAMILTONIAN},
      .num_integrated_diag_moments = 1,
      .integrated_diag_moments = { GKYL_F_MOMENT_HAMILTONIAN },
    },

    .time_rate_diagnostics = true,
  };

  // field
  struct gkyl_gyrokinetic_field field = {
    .poisson_bcs = {
      { .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_FIELD_DIRICHLET, .value = {0.0}, },
      { .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_FIELD_DIRICHLET, .value = {0.0}, },
    },
    .time_rate_diagnostics = true,
  };

  // Geometry
  struct gkyl_efit_inp efit_inp = {
    .rz_poly_order = 2,   // polynomial order for psi(R,Z) used for field line tracing
    .flux_poly_order = 1, // polynomial order for fpol(psi)
  };
  // Copy eqdsk file into efit_inp.
  memcpy(efit_inp.filepath, ctx.eqdsk_file, sizeof(ctx.eqdsk_file));

  struct gkyl_tok_geo_grid_inp grid_inp = {
    .ftype = GKYL_LSN_SOL, // Type of geometry.
    .rclose = 2.5,           // Closest R to region of interest.
    .rright = 2.5,           // Closest R to outboard SOL.
    .rleft = 0.7,            // Closest R to inboard SOL.
    .rmax = 2.5,             // Largest R in machine.
    .rmin = 0.7,             // Smallest R in machine.
    .zmin = -1.3,            // Lower Z boundary.
    .zmax = 1.0,             // Upper Z boundary.
    .zmin_left = -1.0,       // Z of inboard divertor plate.
    .zmin_right = -1.0,      // Z of outboard divertor plate.
  };

  // GK app
  struct gkyl_gk app_inp = {
    .name = "gk_bgk_im_asdex_high_adapt_3x2v_p1",

    .cdim = ctx.cdim,
    .lower = { ctx.x_min, ctx.y_min, ctx.z_min },
    .upper = { ctx.x_max, ctx.y_max, ctx.z_max },
    .cells = { cells_x[0], cells_x[1], cells_x[2] },
    .poly_order = ctx.poly_order,
    .basis_type = app_args.basis_type,
    .cfl_frac = 1.0,

    .geometry = {
      .geometry_id = GKYL_TOKAMAK,
      .efit_info = efit_inp,
      .tok_grid_info = grid_inp,
    },

    .num_periodic_dir = 1,
    .periodic_dirs = {1},

    .num_species = ctx.num_species,
    .species = { elc, ion },

    .field = field,
    .parallelism = {
      .use_gpu = app_args.use_gpu,
      .cuts = {app_args.cuts[0], app_args.cuts[1], app_args.cuts[2]},
      .comm = comm,
    },
  };
  
  struct gkyl_gyrokinetic_run_inp run_inp = {
    .app_inp = app_inp,
    .time_stepping = {
      .t_end = ctx.final_time,
      .num_frames = ctx.num_frames,
      .write_phase_freq = ctx.write_phase_freq,
      .int_diag_calc_num = ctx.int_diag_calc_num,
      .dt_failure_tol = ctx.dt_failure_tol,
      .num_failures_max = ctx.num_failures_max,
      .is_restart = app_args.is_restart,
      .restart_frame = app_args.restart_frame,
      .num_steps = app_args.num_steps,
    }
  };

  gkyl_gyrokinetic_run_simulation(&run_inp);

  gkyl_gyrokinetic_comms_release(comm);

#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi)
    MPI_Finalize();
#endif

  return 0;
}
