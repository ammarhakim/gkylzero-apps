#include <dgc_maxwell.h>

struct sim_ctx {
  double LX, LY;
  int m, n;
};

void
elc_fld(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  struct sim_ctx *sc = ctx;
  double a = sc->m*M_PI/sc->LX;
  double b = sc->n*M_PI/sc->LY;
  double omega = sqrt(a*a+b*b);

  double x = xn[0], y = xn[1];

  fout[0] = 0.0;
  fout[1] = 0.0;
  fout[2] = sin(a*x)*sin(b*y)*cos(omega*t);
}

void
mag_fld(double t, const double* GKYL_RESTRICT xn, double* GKYL_RESTRICT fout, void *ctx)
{
  fout[0] = 0.0;
  fout[1] = 0.0;
  fout[2] = 0.0;
}

int
main(void)
{
  struct sim_ctx sc = {
    .LX = 8.0,
    .LY = 4.0,
    .m = 8,
    .n = 4
  };

  struct dgc_inp inp = {
    .name = "sim_tm_2d",

    .ndim = 2,
    .lower = { 0.0, 0.0 },
    .upper = { sc.LX, sc.LY },
    .cells = { 80, 40 },

    .ctx = &sc,
    .init_E = elc_fld,
    .init_B = mag_fld,
  };

  struct dgc_app *app = dgc_app_new(&inp);

  dgc_app_apply_ics(app);
  dgc_app_write(app, 0.0, 0);

  // compute oscillation frequency
  double a = sc.m*M_PI/sc.LX;
  double b = sc.n*M_PI/sc.LY;
  double omega = sqrt( a*a + b*b );
  
  double tcurr = 0.0, tend = 10.0;
  double dt = dgc_app_max_dt(app);

  dt = fmin(dt, tend-tcurr);

  long step = 1;
  while (tcurr < tend) {
    printf("Taking time-step %ld at t = %g ...", step, tcurr);
    struct gkyl_update_status status = dgc_app_update(app, dt);
    printf(" dt = %g\n", status.dt_actual);

    if (!status.success) {
      fprintf(stderr, "** Update method failed! Aborting simulation ....\n");
      break;
    }
    tcurr += status.dt_actual;

    dt = fmin(tend-tcurr, status.dt_suggested);

    step += 1;
  }
  dgc_app_write(app, tend, 1);

  // force app to initialize with exact solution and write it out for
  // comparison
  dgc_app_reinit(app, tcurr, elc_fld, mag_fld, &sc);
  dgc_app_write(app, tcurr, 1000);

  dgc_app_release(app);

  return 0;
}
