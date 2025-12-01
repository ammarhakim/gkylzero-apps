#include <math.h>
#include <stdio.h>
#include <time.h>

#include <gkyl_const.h>
#include <gkyl_gyrokinetic.h>
#include <gkyl_gyrokinetic_run.h>
#include <gkyl_math.h>

#include <rt_arg_parse.h>

// Define the context of the simulation. This stores global parameters.
struct gk_app_ctx {
    int cdim, vdim;
    double B0;
    // Plasma parameters
    double me, qe, mi, qi, n0, Te0, Ti0;
    // Collision parameters
    double nuFrac, nuElc, nuIon;
    // Source parameters
    int num_sources;
    double lambda_source;
    double x_source;
    // Grid parameters
    char geqdsk_file[128]; // File with equilibrium.
    double Lx, Ly, Lz;
    double x_min, x_max, y_min, y_max, z_min, z_max;
    int num_cell_x, num_cell_y, num_cell_z, num_cell_vpar, num_cell_mu;
    int cells[GKYL_MAX_DIM], poly_order;
    double vpar_max_elc, mu_max_elc, vpar_max_ion, mu_max_ion;
    // Simulation control parameters
    double final_time, write_phase_freq;
    int num_frames, int_diag_calc_num, num_failures_max;
    double dt_failure_tol;
};

void limiter_plate_func_top(double s, double* RZ)
{
  // Tilted, straight plate.
  double RZ_lo_psi[] = {2.456000, 0.749200};
  double RZ_up_psi[] = {2.811277, 0.784605};
  RZ[0] = RZ_lo_psi[0] + (RZ_up_psi[0] - RZ_lo_psi[0])*s;
  RZ[1] = RZ_lo_psi[1] + (RZ_up_psi[1] - RZ_lo_psi[1])*s;
}

void limiter_plate_func_bottom(double s, double* RZ)
{
  // Tilted, straight plate.
  double RZ_lo_psi[] = {2.867071, 0.462058};
  double RZ_up_psi[] = {3.129871, 0.515625};
  RZ[0] = RZ_lo_psi[0] + (RZ_up_psi[0] - RZ_lo_psi[0])*s;
  RZ[1] = RZ_lo_psi[1] + (RZ_up_psi[1] - RZ_lo_psi[1])*s;
}

double polynomial(double x)
{
  return -822.5235171*pow(x, 3.0)	+ 3059.807315*pow(x, 2.0) - 3797.28297*x + 1571.765648;
}

// Density initial condition (like AUG exp profile)
void eval_density(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  double x = xn[0], z = xn[2];
  struct gk_app_ctx *app = ctx;
  double x_min = app->x_min, x_max = app->x_max;
  double rho_min = 1.00168533, rho_max = 1.0160841;
  double rho = (x-x_min)/(x_max-x_min)*(rho_max-rho_min) + rho_min;
  double n0 = app->n0;
  double floor = 0.1*n0;

  fout[0] = n0*polynomial(rho);
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
  double eV = GKYL_ELEMENTARY_CHARGE;
  double x_min = app->x_min, x_max = app->x_max;
  double rho_min = 1.00168533, rho_max = 1.0160841;
  double rho = (x-x_min)/(x_max-x_min)*(rho_max-rho_min) + rho_min;

  fout[0] = (-138.5951825*pow(rho,2) + 163.682554*rho +	1.597206639) * eV;
}

// Ion temperature initial conditions
void eval_temp_ion(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  double x = xn[0], z = xn[2];
  struct gk_app_ctx *app = ctx;
  double eV = GKYL_ELEMENTARY_CHARGE;
  double x_min = app->x_min, x_max = app->x_max;
  double rho_min = 1.00168533, rho_max = 1.0160841;
  double rho = (x-x_min)/(x_max-x_min)*(rho_max-rho_min) + rho_min;

  fout[0] = (-138.5951825*pow(rho,2) + 163.682554*rho +	1.597206639) * eV;
}

// Source density
void
eval_density_source(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  struct gk_app_ctx *app = ctx;
  double x = xn[0], y = xn[1], z = xn[2];
  double lambda_source = app->lambda_source;
  double x_source = app->x_source;
  double Lz = app->Lz;
  double z_source = -Lz/4.0;
  double S0 = 6.0e22;
  double source_floor = 0.1*S0;

  if (fabs(z-z_source)<Lz/8.0)
    fout[0] = fmax(S0*exp(-(x-x_source)*(x-x_source)/(2*lambda_source*lambda_source)), source_floor);
  else
    fout[0] = source_floor;
}

// Source parallel velocity
void
eval_upar_source(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  fout[0] = 0.0;
}

// Electron source temperature
void
eval_temp_elc_source(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  struct gk_app_ctx *app = ctx;
  double x = xn[0], y = xn[1], z = xn[2];
  double lambda_source = app->lambda_source;
  double x_source = app->x_source;
  double Lz = app->Lz;
  double z_source = -Lz/4.0;
  double eV = GKYL_ELEMENTARY_CHARGE;
  if ((x < x_source + 3*lambda_source) && (fabs(z-z_source)<Lz/8.0))
    fout[0] = 62.5*eV;
  else
    fout[0] = 5.711832032857547*eV;
}

// Ion source temperature
void
eval_temp_ion_source(double t, const double * GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  struct gk_app_ctx *app = ctx;
  double x = xn[0], y = xn[1], z = xn[2];
  double lambda_source = app->lambda_source;
  double x_source = app->x_source;
  double Lz = app->Lz;
  double z_source = -Lz/4.0;
  double eV = GKYL_ELEMENTARY_CHARGE;
  if ((x < x_source + 3*lambda_source) && (fabs(z-z_source)<Lz/8.0))
    fout[0] = 94.5*eV;
  else
    fout[0] = 15.05699091505553*eV;
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
  double B0 = 0.5*(3.062641e+00+4.484922e+00);
  // Universal constant parameters.
  double eps0 = GKYL_EPSILON0, eV = GKYL_ELEMENTARY_CHARGE;
  double mp = GKYL_PROTON_MASS, me = GKYL_ELECTRON_MASS;
  double qi = eV; // ion charge
  double qe = -eV; // electron charge

  // Plasma parameters
  double AMU = 2.01410177811;
  double mi  = mp*AMU;   // Deuterium ions
  double Te0 = 62.5*eV;
  double Ti0 = 94.5*eV;
  double n0  = 1.0e18;   // [1/m^3]
  double vte = sqrt(Te0/me), vti = sqrt(Ti0/mi); // Thermal velocities
  double c_s = sqrt(Te0/mi);
  double omega_ci = fabs(qi*B0/mi);
  double rho_s = c_s/omega_ci;

  // Location of the numerical equilibrium.
  char geqdsk_file[128] = "west_LSN_zeroed.geqdsk";

  // Get the separatrix psi, location of the X-point, and magnetic axis coords.
  struct gkyl_efit_inp efit_inp = {
    // psiRZ and related inputs
    .rz_poly_order = 1,
    .flux_poly_order = 1,
  };
  // Copy eqdsk file into efit_inp.
  memcpy(efit_inp.filepath, geqdsk_file, sizeof(geqdsk_file));
  struct gkyl_efit *efit = gkyl_efit_new(&efit_inp);
  // Psi at separatrix (LCFS): 0.4167022815
  double psi_sep = 0.4167022815; //efit->psisep;
  double Rxpt = efit->Rxpt[0], Zxpt = efit->Zxpt[0];
  double R_axis = efit->rmaxis, Z_axis = efit->zmaxis;
  gkyl_efit_release(efit);

  // Configuration domain parameters 
//  double x_min = 0.403;
//  double x_max = 0.4165;
  double x_min = 0.35;
  double x_max = 0.36;
  double Lx = x_max - x_min;

  printf("psi_sep = %.9e\n",psi_sep);
  printf("psi_min = %.9e\n",x_min);
  printf("psi_max = %.9e\n",x_max);
  printf("rho_norm_min = %.9g\n",(psi_sep+fabs(x_min-psi_sep))/psi_sep);
  printf("rho_norm_max = %.9g\n",(psi_sep+fabs(x_max-psi_sep))/psi_sep);

  double R0 = 2.978; // Estimated visually.
  double r0 = R0-R_axis;
  double q0 = 2.44; // Estimated internally in gkeyll. 
  double Ly = 100*rho_s*q0/r0;
  double y_min = -Ly/2.;
  double y_max =  Ly/2.;

  double Lz    = 2.*M_PI-1e-10;       // Domain size along magnetic field.
  double z_min = -Lz/2.;
  double z_max =  Lz/2.;

  // Collision frequencies
  double nuFrac = 0.5;
  // Electron-electron collision freq.
  double logLambdaElc = 6.6 - 0.5 * log(n0/1e20) + 1.5 * log(Ti0/eV);
  double nuElc = nuFrac * logLambdaElc * pow(eV, 4) * n0 /
    (6*sqrt(2.) * pow(M_PI,3./2.) * pow(eps0,2) * sqrt(me) * pow(Te0,3./2.));
  // Ion-ion collision freq.
  double logLambdaIon = 6.6 - 0.5 * log(n0/1e20) + 1.5 * log(Ti0/eV);
  double nuIon = nuFrac * logLambdaIon * pow(eV, 4) * n0 /
    (12 * pow(M_PI,3./2.) * pow(eps0,2) * sqrt(mi) * pow(Ti0,3./2.));

  // Source parameters
  int num_sources = 1;
  double x_source = 0.084;
  double lambda_source = 0.001;

  // Grid parameters
  int num_cell_x = 4; 
  int num_cell_y = 4;
  int num_cell_z = 8;
  int num_cell_vpar = 8;
  int num_cell_mu = 4;
  int poly_order = 1;
  // Velocity box dimensions
  double vpar_max_elc = 6.*vte;
  double mu_max_elc   = me*pow(4*vte,2)/(2*B0);
  double vpar_max_ion = 6.*vti;
  double mu_max_ion   = mi*pow(4*vti,2)/(2*B0);
  double final_time = 100.e-6;
  int num_frames = 10;
  double write_phase_freq = 0.01;
  int int_diag_calc_num = num_frames*100;
  double dt_failure_tol = 1.0e-3; // Minimum allowable fraction of initial time-step.
  int num_failures_max = 20; // Maximum allowable number of consecutive small time-steps.

  struct gk_app_ctx ctx = {
    .cdim = cdim,
    .vdim = vdim,
    .B0     = B0    ,
    .Lx     = Lx    ,
    .Ly     = Ly    ,
    .Lz     = Lz    ,
    .x_min = x_min,  .x_max = x_max,
    .y_min = y_min,  .y_max = y_max,
    .z_min = z_min,  .z_max = z_max,
    .me = me,  .qe = qe,
    .mi = mi,  .qi = qi,
    .n0 = n0,  .Te0 = Te0,  .Ti0 = Ti0,
    .nuFrac = nuFrac,  .nuElc = nuElc,  .nuIon = nuIon,
    .num_sources = num_sources,
    .lambda_source = lambda_source,  
    .x_source = x_source,
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
  memcpy(ctx.geqdsk_file, geqdsk_file, sizeof(geqdsk_file));

  return ctx;
}

int 
main(int argc, char **argv)
{
  struct gkyl_app_args app_args = parse_app_args(argc, argv);

#ifdef GKYL_HAVE_MPI
  if (app_args.use_mpi)
    MPI_Init(&argc, &argv);
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

  // Electrons.
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

    .correct = {
      .correct_all_moms = true,
      .use_last_converged = true,
      .iter_eps = 1e-12,
      .max_iter = 10,
    },

    .collisionless = {
      .type = GKYL_GK_COLLISIONLESS_ES,
    },

    .collisions =  {
      .collision_id = GKYL_BGK_COLLISIONS,
      .den_ref = ctx.n0, // Density used to calculate coulomb logarithm
      .temp_ref = ctx.Te0, // Temperature used to calculate coulomb logarithm
      .is_implicit = true,
      .num_cross_collisions = 1,
      .collide_with = { "ion" },
      .write_diagnostics = true,
    },

    .source = {
      .source_id = GKYL_PROJ_SOURCE,
      .num_sources = ctx.num_sources,
      .projection[0] = {
        .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
        .ctx_density = &ctx,
        .density = eval_density_source,
        .ctx_upar = &ctx,
        .upar= eval_upar_source,
        .ctx_temp = &ctx,
        .temp = eval_temp_elc_source,
      },
    },

    .bcs = {
      { .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB, },
      { .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB, },
      { .dir = 2, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_SHEATH, },
      { .dir = 2, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_SHEATH, },
    },

    .num_diag_moments = 10,
    .diag_moments = {GKYL_F_MOMENT_HAMILTONIAN, GKYL_F_MOMENT_MAXWELLIAN, GKYL_F_MOMENT_BIMAXWELLIAN, GKYL_F_MOMENT_M0, GKYL_F_MOMENT_M1, GKYL_F_MOMENT_M2PAR, GKYL_F_MOMENT_M2PERP, GKYL_F_MOMENT_M2, GKYL_F_MOMENT_M3PAR, GKYL_F_MOMENT_M3PERP},
    .num_integrated_diag_moments = 1,
    .integrated_diag_moments = { GKYL_F_MOMENT_HAMILTONIAN },
    .boundary_flux_diagnostics = {
      .num_integrated_diag_moments = 1,
      .integrated_diag_moments = { GKYL_F_MOMENT_HAMILTONIAN },
    },
  };

  // Ions.
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
      .is_implicit = true,
      .num_cross_collisions = 1,
      .collide_with = { "elc" },
      .write_diagnostics = true,
    },

    .source = {
      .source_id = GKYL_PROJ_SOURCE,
      .num_sources = ctx.num_sources,
      .projection[0] = {
        .proj_id = GKYL_PROJ_MAXWELLIAN_PRIM,
        .ctx_density = &ctx,
        .density = eval_density_source,
        .ctx_upar = &ctx,
        .upar= eval_upar_source,
        .ctx_temp = &ctx,
        .temp = eval_temp_ion_source,
      },
    },

    .bcs = {
      { .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB, },
      { .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_ABSORB, },
      { .dir = 2, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_SPECIES_SHEATH, },
      { .dir = 2, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_SPECIES_SHEATH, },
    },

    .num_diag_moments = 10,
    .diag_moments = {GKYL_F_MOMENT_HAMILTONIAN, GKYL_F_MOMENT_MAXWELLIAN, GKYL_F_MOMENT_BIMAXWELLIAN, GKYL_F_MOMENT_M0, GKYL_F_MOMENT_M1, GKYL_F_MOMENT_M2PAR, GKYL_F_MOMENT_M2PERP, GKYL_F_MOMENT_M2, GKYL_F_MOMENT_M3PAR, GKYL_F_MOMENT_M3PERP},
    .num_integrated_diag_moments = 1,
    .integrated_diag_moments = { GKYL_F_MOMENT_HAMILTONIAN },
    .boundary_flux_diagnostics = {
      .num_integrated_diag_moments = 1,
      .integrated_diag_moments = { GKYL_F_MOMENT_HAMILTONIAN },
    },
  };

  // Field.
  struct gkyl_gyrokinetic_field field = {
    .poisson_bcs = {
      { .dir = 0, .edge = GKYL_LOWER_EDGE, .type = GKYL_BC_GK_FIELD_DIRICHLET, .value = {0.0}, },
      { .dir = 0, .edge = GKYL_UPPER_EDGE, .type = GKYL_BC_GK_FIELD_DIRICHLET, .value = {0.0}, },
    },
  };

  // Geometry
  struct gkyl_efit_inp efit_inp = {
    // psiRZ and related inputs
    .rz_poly_order = 2,          // polynomial order for psi(R,Z) used for field line tracing
    .flux_poly_order = 1,        // polynomial order for fpol(psi)
  };
  // Copy eqdsk file into efit_inp.
  memcpy(efit_inp.filepath, ctx.geqdsk_file, sizeof(ctx.geqdsk_file));

  struct gkyl_tok_geo_grid_inp grid_inp = {
    .ftype = GKYL_LSN_SOL,    // Type of geometry.
    .rclose = 3.1,           // Closest R to region of interest.
    .rleft = 2.4,            // Closest R to inboard SOL.
    .rright = 3.1,           // Closest R to outboard SOL.
    .rmin = 2.4,             // Smallest R in machine.
    .rmax = 3.1,             // Largest R in machine.
    .zmin = 0.4,            // Lower Z boundary.
    .zmax =  0.8,             // Upper Z boundary.
    .zmin_left = 0.8,       // Z of inboard divertor plate.
    .zmin_right = 0.4,      // Z of outboard divertor plate.
    .plate_spec = true,
    .plate_func_lower = limiter_plate_func_bottom,
    .plate_func_upper = limiter_plate_func_top,
  };

  // Parallelism
  struct gkyl_app_parallelism_inp parallelism = {
    .comm = comm,
    .cuts = {app_args.cuts[0], app_args.cuts[1], app_args.cuts[2]},
    .use_gpu = app_args.use_gpu,
  };

  // GK app
  struct gkyl_gk app_inp = {
    .name = "gk_west_lsn_sol_3x2v_p1",

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

    .num_species = 2,
    .species = { elc, ion },

    .field = field,
    .parallelism = parallelism
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
