#pragma once

#include <gkylzero.h>

typedef struct dgc_app dgc_app;

// Input parameters to app
struct dgc_inp {
  char name[128]; // name of app: used as output prefix
  
  int ndim; // dimension
  double lower[3], upper[3]; // lower, upper bounds
  int cells[3]; // config-space cells

  double cfl_frac; // CFL fraction to use

  // initial condition for electric and magnetic field
  evalf_t init_E, init_B;
  void *ctx; // eval context
};

// Create new app object
dgc_app *dgc_app_new(const struct dgc_inp *inp);

// Initialize the simulation with initial conditions
void dgc_app_apply_ics(dgc_app *app);

// Reinitialize the simulation using supplied functions
void dgc_app_reinit(dgc_app *app, double tm, evalf_t efunc, evalf_t bfunc, void *ctx);

// Write to file
void dgc_app_write(dgc_app *app, double tm, int frame);

// Write EM energy to file
void dgc_app_write_em_energy(dgc_app *app);

// Maximum stable time-step
double dgc_app_max_dt(const dgc_app *app);

// Take a time-step
struct gkyl_update_status dgc_app_update(dgc_app *app, double dt);

// compute EM energy
void dgc_app_calc_em_energy(const dgc_app *app, double tm);

// Free memory used in app
void dgc_app_release(dgc_app *app);
