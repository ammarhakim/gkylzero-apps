#include <dgc_maxwell.h>

// ranges for use in BCs
struct app_skin_ghost_ranges {
  struct gkyl_range lower_skin[GKYL_MAX_DIM];
  struct gkyl_range lower_ghost[GKYL_MAX_DIM];

  struct gkyl_range upper_skin[GKYL_MAX_DIM];
  struct gkyl_range upper_ghost[GKYL_MAX_DIM];
};

// multivector in 3D GA
struct mv_array {
  struct gkyl_array *m0; // scalar
  struct gkyl_array *m1; // vector
  struct gkyl_array *m2; // bivectors
  struct gkyl_array *ps; // pseudoscalar
};

// Create ghost and skin sub-ranges given a parent range
static void
skin_ghost_ranges_init(struct app_skin_ghost_ranges *sgr,
  const struct gkyl_range *parent, const int *ghost)
{
  int ndim = parent->ndim;
  
  for (int d=0; d<ndim; ++d) {
    gkyl_skin_ghost_ranges(&sgr->lower_skin[d], &sgr->lower_ghost[d],
      d, GKYL_LOWER_EDGE, parent, ghost);
    gkyl_skin_ghost_ranges(&sgr->upper_skin[d], &sgr->upper_ghost[d],
      d, GKYL_UPPER_EDGE, parent, ghost);
  }
}

static struct mv_array*
mv_array_new(size_t sz)
{
  struct mv_array *mva = gkyl_malloc(sizeof(*mva));

  mva->m0 = gkyl_array_new(GKYL_DOUBLE, 1, sz);
  gkyl_array_clear(mva->m0, 0.0);

  mva->m1 = gkyl_array_new(GKYL_DOUBLE, 3, sz);
  gkyl_array_clear(mva->m1, 0.0);

  mva->m2 = gkyl_array_new(GKYL_DOUBLE, 3, sz);
  gkyl_array_clear(mva->m2, 0.0);

  mva->ps = gkyl_array_new(GKYL_DOUBLE, 1, sz);
  gkyl_array_clear(mva->ps, 0.0);

  return mva;
}

// Compute out = c1*arr1 + c2*arr2
static inline struct gkyl_array*
array_combine(struct gkyl_array *out, double c1, const struct gkyl_array *arr1,
  double c2, const struct gkyl_array *arr2, const struct gkyl_range rng)
{
  return gkyl_array_accumulate_range(gkyl_array_set_range(out, c1, arr1, rng),
    c2, arr2, rng);
}

// Compute out = c1*arr1 + c2*arr2
static struct mv_array*
mv_array_combine(struct mv_array *out, double c1, const struct mv_array *arr1,
  double c2, const struct mv_array *arr2, const struct gkyl_range rng)
{
  array_combine(out->m0, c1, arr1->m0, c2, arr2->m0, rng);
  array_combine(out->m1, c1, arr1->m1, c2, arr2->m1, rng);
  array_combine(out->m2, c1, arr1->m2, c2, arr2->m2, rng);
  array_combine(out->ps, c1, arr1->ps, c2, arr2->ps, rng);

  return out;
}

void
mv_array_release(struct mv_array *mva)
{
  gkyl_array_release(mva->m0);
  gkyl_array_release(mva->m1);
  gkyl_array_release(mva->m2);
  gkyl_array_release(mva->ps);

  gkyl_free(mva);
}

// labels for cells involved in FD stencil
enum offset_labels {
  I, // current cell
  MII, PII, // left/right
  IMI, IPI, // bottom/top
  IIM, IIP, // back/front
  OFF_END
};

// offsets for cells involved in FD stencil
static int offset_idx[][3] = {
  [I] = { 0, 0, 0 },
  [MII] = { -1, 0, 0 },
  [PII] = { 1, 0, 0 },
  [IMI] = { 0, -1, 0 },
  [IPI] = { 0, 1, 0 },
  [IIM] = { 0, 0, -1 },
  [IIP] = { 0, 0, 1 }
};

// compute cell offsets
void
calc_offsets(const struct gkyl_range *range, long offsets[OFF_END])
{
  for (int i=0; i<OFF_END; ++i)
    offsets[i] = gkyl_range_offset(range, offset_idx[i]);
}
 
// Discrete GC Maxwell solver app
struct dgc_app {
  char name[128]; // name of app  
  int ndim;
  double tcurr;
  double cfl;

  bool is_first; // first call to update
  
  struct gkyl_rect_grid grid;
  struct gkyl_range local, local_ext;

  struct mv_array *Fhalf_new, *Fhalf;
  struct mv_array *Ffull_new, *Ffull;
  struct mv_array *gradf;

  bool is_first_energy_write_call;
  gkyl_dynvec em_energy;

  struct gkyl_comm *comm;

  evalf_t init_E, init_B;
  void *ctx;
};

struct dgc_app*
dgc_app_new(const struct dgc_inp *inp)
{
  struct dgc_app *app = gkyl_malloc(sizeof(*app));

  strcpy(app->name, inp->name);
  int ndim = app->ndim = inp->ndim;

  app->tcurr = 0.0;
  double cfl_frac = inp->cfl_frac > 0 ? inp->cfl_frac : 1.0;
  app->cfl = cfl_frac/ndim; // this does not seem correct
  
  app->is_first = true;

  gkyl_rect_grid_init(&app->grid, inp->ndim, inp->lower, inp->upper, inp->cells);

  int nghost[GKYL_MAX_CDIM] = { 1, 1, 1 };
  gkyl_create_grid_ranges(&app->grid, nghost, &app->local_ext, &app->local);

  app->Ffull = mv_array_new(app->local_ext.volume);
  app->Ffull_new = mv_array_new(app->local_ext.volume);
  app->Fhalf = mv_array_new(app->local_ext.volume);
  app->Fhalf_new = mv_array_new(app->local_ext.volume);
  app->gradf = mv_array_new(app->local_ext.volume);

  app->is_first_energy_write_call = true;
  app->em_energy = gkyl_dynvec_new(GKYL_DOUBLE, 6);

  app->init_E = inp->init_E;
  app->init_B = inp->init_B;
  app->ctx = inp->ctx;

  int cuts[GKYL_MAX_CDIM] = { 1, 1, 1 };
  struct gkyl_rect_decomp *decomp = gkyl_rect_decomp_new_from_cuts(ndim,
    cuts, &app->local);

  app->comm = gkyl_null_comm_inew( &(struct gkyl_null_comm_inp) {
      .decomp = decomp
    }
  );

  gkyl_rect_decomp_release(decomp);

  return app;
}

static void
apply_bc(const struct dgc_app *app, struct mv_array *mv_arr)
{
  int per_dirs[] = { 0, 1, 2 };
  gkyl_comm_array_per_sync(app->comm, &app->local, &app->local_ext,
    app->ndim, per_dirs, mv_arr->m0);
  gkyl_comm_array_per_sync(app->comm, &app->local, &app->local_ext,
    app->ndim, per_dirs, mv_arr->m1);
  gkyl_comm_array_per_sync(app->comm, &app->local, &app->local_ext,
    app->ndim, per_dirs, mv_arr->m2);
  gkyl_comm_array_per_sync(app->comm, &app->local, &app->local_ext,
    app->ndim, per_dirs, mv_arr->ps);  
}

void
dgc_app_apply_ics(struct dgc_app *app)
{
  dgc_app_reinit(app, 0.0, app->init_E, app->init_B, app->ctx);
}

void
dgc_app_reinit(dgc_app *app, double tm, evalf_t efunc, evalf_t bfunc, void *ctx)
{
  app->tcurr = tm;
  
  // Initialize electric field
  struct gkyl_offset_descr offset_E[3] = {
    { 0.0, -0.5, -0.5 },
    { -0.5, 0.0, -0.5 },
    { -0.5, -0.5, 0.0 }
  };
  gkyl_eval_offset_fd *ev_E = gkyl_eval_offset_fd_new(
    &(struct gkyl_eval_offset_fd_inp) {
      .grid = &app->grid,
      .num_ret_vals = 3,
      .eval = efunc,
      .ctx = ctx,
      .offsets = offset_E
    }
  );
  gkyl_eval_offset_fd_advance(ev_E, tm, &app->local, app->Ffull->m1);
  gkyl_eval_offset_fd_release(ev_E);

  // Initialize magnetic field
  struct gkyl_offset_descr offset_B[3] = {
    { -0.5, 0.0, 0.0 },
    { 0.0, -0.5, 0.0 },
    { 0.0, 0.0, -0.5 }
  };
  gkyl_eval_offset_fd *ev_B = gkyl_eval_offset_fd_new(
    &(struct gkyl_eval_offset_fd_inp){
      .grid = &app->grid,
      .eval = bfunc,
      .num_ret_vals = 3,
      .ctx = ctx,
      .offsets = offset_B
    }
  );
  gkyl_eval_offset_fd_advance(ev_B, tm, &app->local, app->Ffull->m2);
  gkyl_eval_offset_fd_release(ev_B);

  apply_bc(app, app->Ffull);  
}

void
dgc_app_write(struct dgc_app *app, double tm, int frame)
{
  char fileNm[1024]; // buffer for file name
  
  do {
    const char *fmt = "%s_m0_%d.gkyl";
    snprintf(fileNm, sizeof fileNm, fmt, app->name, frame);
    
    gkyl_grid_sub_array_write(&app->grid, &app->local, app->Ffull->m0, fileNm);
  } while (0);

  do {
    const char *fmt = "%s_m1_%d.gkyl";
    snprintf(fileNm, sizeof fileNm, fmt, app->name, frame);

    gkyl_grid_sub_array_write(&app->grid, &app->local, app->Ffull->m1, fileNm);
  } while (0);

  do {
    const char *fmt = "%s_m2_%d.gkyl";
    snprintf(fileNm, sizeof fileNm, fmt, app->name, frame);

    gkyl_grid_sub_array_write(&app->grid, &app->local, app->Ffull->m2, fileNm);
  } while (0);

  do {
    const char *fmt = "%s_ps_%d.gkyl";
    snprintf(fileNm, sizeof fileNm, fmt, app->name, frame);

    gkyl_grid_sub_array_write(&app->grid, &app->local, app->Ffull->ps, fileNm);
  } while (0);
}

void
dgc_app_write_em_energy(dgc_app *app)
{
  char fileNm[1024]; // buffer for file name  
  const char *fmt = "%s_em_energy.gkyl";
  snprintf(fileNm, sizeof fileNm, fmt, app->name);
    
  if (app->is_first_energy_write_call) {
    gkyl_dynvec_write(app->em_energy, fileNm);
    app->is_first_energy_write_call = false;
  }
  else {
    gkyl_dynvec_awrite(app->em_energy, fileNm);
  }
}

double
dgc_app_max_dt(const dgc_app *app)
{
  double c = 1.0; // light-speed
  double omega_cfl = 0.0;
  
  for (int d=0; d<app->ndim; ++d)
    omega_cfl += c/app->grid.dx[d];
  
  return app->cfl/omega_cfl;
}

static void
mv_array_copy(struct dgc_app *app, struct mv_array *fout, const struct mv_array *finp)
{
  gkyl_array_copy(fout->m0, finp->m0);
  gkyl_array_copy(fout->m1, finp->m1);
  gkyl_array_copy(fout->m2, finp->m2);
  gkyl_array_copy(fout->ps, finp->ps);
}

// compute gradient of multivector f and store output in gradf
static void
calc_grad_f(const struct dgc_app *app, const struct mv_array *f, struct mv_array *gradf)
{
  int ndim = app->grid.ndim;
  enum { IX1, IX2, IX3 }; // indexing

  double dx[3] = {
    app->grid.dx[0], app->grid.dx[1], app->grid.dx[3]
  };
  
  long offsets[OFF_END];
  calc_offsets(&app->local, offsets);

  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, &app->local);
  
  while ( gkyl_range_iter_next(&iter) ) {
    long lidx = gkyl_range_idx(&app->local, iter.idx);

    const double *alpha_I = gkyl_array_cfetch(f->m0, lidx+offsets[I]);
    const double *a_I = gkyl_array_cfetch(f->m1, lidx+offsets[I]);    
    const double *b_I = gkyl_array_cfetch(f->m2, lidx+offsets[I]);
    const double *beta_I = gkyl_array_cfetch(f->ps, lidx+offsets[I]);
    
    // grad(f) grade-0
    double *gf0 = gkyl_array_fetch(gradf->m0, lidx);
    gf0[0] = 0.0;

    if (ndim > 0) {
      const double *a_MII = gkyl_array_cfetch(f->m1, lidx+offsets[MII]);
      gf0[0] +=  (a_I[0]-a_MII[0])/dx[0];
    }
    if (ndim > 1) {
      const double *a_IMI = gkyl_array_cfetch(f->m1, lidx+offsets[IMI]);
      gf0[0] +=  (a_I[1]-a_IMI[1])/dx[1];
    }
    if (ndim > 2) {
      const double *a_IIM = gkyl_array_cfetch(f->m1, lidx+offsets[IIM]);
      gf0[0] +=  (a_I[2]-a_IIM[2])/dx[2];
    }

    // grad(f) grade-1
    double *gf1 = gkyl_array_fetch(gradf->m1, lidx);
    gf1[0] = gf1[1] = gf1[2] = 0.0;

    if (ndim > 0) {
      const double *alpha_PII = gkyl_array_cfetch(f->m0, lidx+offsets[PII]);
      const double *b_MII = gkyl_array_cfetch(f->m2, lidx+offsets[MII]);
      
      gf1[IX1] += (alpha_PII[0]-alpha_I[0])/dx[0];
      
      gf1[IX3] += -(b_I[IX2]-b_MII[IX2])/dx[0];
      gf1[IX2] += (b_I[IX3]-b_MII[IX3])/dx[0];
    }
    if (ndim > 1) {
      const double *alpha_IPI = gkyl_array_cfetch(f->m0, lidx+offsets[IPI]);
      const double *b_IMI = gkyl_array_cfetch(f->m2, lidx+offsets[IMI]);
      
      gf1[IX2] +=  (alpha_IPI[0]-alpha_I[0])/dx[1];
      
      gf1[IX3] += (b_I[IX1]-b_IMI[IX1])/dx[1];
      gf1[IX1] += -(b_I[IX3]-b_IMI[IX3])/dx[1];

    }
    if (ndim > 2) {
      const double *alpha_IIP = gkyl_array_cfetch(f->m0, lidx+offsets[IIP]);
      const double *b_IIM = gkyl_array_cfetch(f->m2, lidx+offsets[IIM]);
      
      gf1[IX3] +=  (alpha_IIP[0]-alpha_I[0])/dx[2];

      gf1[IX1] += (b_I[IX2]-b_IIM[IX2])/dx[2];
      gf1[IX2] += -(b_I[IX1]-b_IIM[IX1])/dx[2];
    }

    // grad(f) grade-2
    double *gf2 = gkyl_array_fetch(gradf->m2, lidx);
    gf2[0] = gf2[1] = gf2[2] = 0.0;

    if (ndim > 0) {
      const double *beta_MII = gkyl_array_cfetch(f->ps, lidx+offsets[MII]);
      const double *a_PII = gkyl_array_cfetch(f->m1, lidx+offsets[PII]);
      
      gf2[IX1] += (beta_I[0]-beta_MII[0])/dx[0];
      
      gf2[IX3] += (a_PII[IX2]-a_I[IX2])/dx[0];
      gf2[IX2] += -(a_PII[IX3]-a_I[IX3])/dx[0];
    }
    if (ndim > 1) {
      const double *beta_IMI = gkyl_array_cfetch(f->ps, lidx+offsets[IMI]);
      const double *a_IPI = gkyl_array_cfetch(f->m1, lidx+offsets[IPI]);
      
      gf2[IX2] += (beta_I[0]-beta_IMI[0])/dx[1];

      gf2[IX1] += (a_IPI[IX3]-a_I[IX3])/dx[1];
      gf2[IX3] += -(a_IPI[IX1]-a_I[IX1])/dx[1];
      
    }
    if (ndim > 2) {
      const double *beta_IIM = gkyl_array_cfetch(f->ps, lidx+offsets[IIM]);
      const double *a_IIP = gkyl_array_cfetch(f->m1, lidx+offsets[IIP]);
      
      gf2[IX3] += (beta_I[0]-beta_IIM[0])/dx[2];

      gf2[IX2] += (a_IIP[IX1]-a_I[IX1])/dx[2];
      gf2[IX1] += -(a_IIP[IX2]-a_I[IX2])/dx[2];
    }    

    // grad(f) pseudoscalar
    double *gfps = gkyl_array_fetch(gradf->ps, lidx);
    gfps[0] = 0.0;

    if (ndim > 0) {
      const double *b_PII = gkyl_array_cfetch(f->m2, lidx+offsets[PII]);
      gfps[0] = (b_PII[0]-b_I[0])/dx[0];
    }
    if (ndim > 1) {
      const double *b_IPI = gkyl_array_cfetch(f->m2, lidx+offsets[IPI]);
      gfps[0] +=  (b_IPI[1]-b_I[1])/dx[1];
    }
    if (ndim > 2) {
      const double *b_IIP = gkyl_array_cfetch(f->m2, lidx+offsets[IIP]);
      gfps[0] +=  (b_IIP[2]-b_I[2])/dx[2];
    }
  }
}

static struct gkyl_update_status
leap_frog(dgc_app *app, double dt)
{
  double dt_max = dgc_app_max_dt(app);
  // use user supplied dt if it is sufficiently small
  double dta = dt < dt_max ? dt : dt_max;

  if (app->is_first) {
    // need to compute Fhalf using ICs before taking step
    calc_grad_f(app, app->Ffull, app->gradf);
    mv_array_combine(app->Fhalf, 1.0, app->Ffull, -dta/2, app->gradf, app->local);
    apply_bc(app, app->Fhalf);
    
    app->is_first = false;
  }

  // update full-step solution
  calc_grad_f(app, app->Fhalf, app->gradf);
  mv_array_combine(app->Ffull_new, 1.0, app->Ffull, -dta, app->gradf, app->local);
  apply_bc(app, app->Ffull_new);
  
  // update half-step solution
  calc_grad_f(app, app->Ffull_new, app->gradf);
  mv_array_combine(app->Fhalf_new, 1.0, app->Fhalf, -dta, app->gradf, app->local);
  apply_bc(app, app->Fhalf_new);

  // copy solution over
  mv_array_copy(app, app->Ffull, app->Ffull_new);
  mv_array_copy(app, app->Fhalf, app->Fhalf_new);
  
  return (struct gkyl_update_status) {
    .dt_actual = dta,
    .dt_suggested = dt_max,
    .success = true    
  };
}

struct gkyl_update_status
dgc_app_update(dgc_app *app, double dt)
{
  struct gkyl_update_status status = leap_frog(app, dt);
  app->tcurr += status.dt_actual;

  return status;
}

void
dgc_app_calc_em_energy(const dgc_app *app, double tm)
{
#define SQ(x) ((x)*(x))

  double vol = app->grid.cellVolume;
  
  struct gkyl_range_iter iter;
  gkyl_range_iter_init(&iter, &app->local);

  double em_energy[6] = { 0.0 };
  
  while ( gkyl_range_iter_next(&iter) ) {
    long lidx = gkyl_range_idx(&app->local, iter.idx);

    const double *a_I = gkyl_array_cfetch(app->Ffull->m1, lidx);
    const double *b_I = gkyl_array_cfetch(app->Ffull->m2, lidx);

    em_energy[0] += SQ(a_I[0]);
    em_energy[1] += SQ(a_I[1]);
    em_energy[2] += SQ(a_I[2]);

    em_energy[3] += SQ(b_I[0]);
    em_energy[4] += SQ(b_I[1]);
    em_energy[5] += SQ(b_I[2]);
  }

  for (int c=0; c<6; ++c) em_energy[c] *= vol;
  gkyl_dynvec_append(app->em_energy, tm, em_energy);
  
#undef SQ
}

void
dgc_app_release(struct dgc_app *app)
{
  mv_array_release(app->Fhalf);
  mv_array_release(app->Fhalf_new);
  mv_array_release(app->Ffull);
  mv_array_release(app->Ffull_new);
  mv_array_release(app->gradf);

  gkyl_comm_release(app->comm);
  gkyl_dynvec_release(app->em_energy);
  
  gkyl_free(app);
}
