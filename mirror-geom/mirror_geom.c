#include <gkyl_alloc.h>
#include <gkyl_array_rio.h>
#include <gkyl_basis.h>
#include <gkyl_dg_basis_ops.h>
#include <gkyl_math.h>
#include <gkyl_mirror_geom.h>
#include <gkyl_rect_decomp.h>

// context for use in root finder
struct psirz_ctx {
  struct gkyl_basis_ops_evalf *evcub; // cubic eval functions
  double Z; // local Z value
  double psi; // psi to match
};

static
double psirz(double R, void *ctx)
{
  struct psirz_ctx *rctx = ctx;
  double Z = rctx->Z;
  double xn[2] = { R, Z };
  double fout[1];
  rctx->evcub->eval_cubic(0, xn, fout, rctx->evcub->ctx);
  return fout[0] - rctx->psi;
}

struct gkyl_mirror_geom *
gkyl_mirror_geom_inew(const struct gkyl_mirror_geom_inp *inp)
{
  struct gkyl_mirror_geom *geo = gkyl_malloc(sizeof *geo);

  int nr = inp->nrnodes, nz = inp->nznodes;
  int cells[] = { nr, nz };
  double lower[2] = { inp->R[0], inp->Z[0] };
  double upper[2] = { inp->R[1], inp->Z[1] };

  struct gkyl_rect_grid gridRZ;
  gkyl_rect_grid_init(&gridRZ, 2, lower, upper, cells);

  struct gkyl_basis_ops_evalf *evcub =
    gkyl_dg_basis_ops_evalf_new(&gridRZ, inp->psiRZ);

  do {
    const char *fname = inp->psi_cubic_fname ? inp->psi_cubic_fname : "psi_cubic.gkyl";
    if (inp->write_psi_cubic)
      gkyl_dg_basis_ops_evalf_write_cubic(evcub, fname);
  } while (0);

  // construct grid in RZ plane
  
  enum { NPSI, NZ };
  int nc[2];
  nc[NPSI] = inp->comp_grid->cells[0]+1;
  nc[NZ] = inp->comp_grid->cells[2]+1;
  
  long nctot = nc[NPSI]*nc[NZ];
  geo->nodesrz = gkyl_array_new(GKYL_DOUBLE, 2, nctot);

  struct gkyl_range node_rng;
  gkyl_range_init_from_shape(&node_rng, 2, nc);

  double zlow = inp->comp_grid->lower[2], zup = inp->comp_grid->upper[2];
  double dz = (zup-zlow)/(nc[NZ]-1);

  double psi_lo = inp->comp_grid->lower[0];
  double psi_up = inp->comp_grid->upper[0];

  // adjust if we are using sqrt(psi) as radial coordinate
  double psic_lo = psi_lo, psic_up = psi_up;
  if (inp->fl_coord == GKYL_MIRROR_GEOM_SQRT_PSI_CART_Z) {
    psic_lo = sqrt(psi_lo);
    psic_up = sqrt(psi_up);
  }
  
  double dpsi = (psic_up-psic_lo)/(nc[NPSI]-1);

  double rlow = lower[0], rup = upper[0];
  double rmin = rlow + 1e-6*(rup-rlow); // avoid going exactly to zero for now

  struct psirz_ctx pctx = { .evcub = evcub };

  bool status = true;
  for (int iz=0; iz<nc[NZ]; ++iz) {
    double zcurr = zlow + iz*dz;

    double psi_min[1], psi_max[1];
    evcub->eval_cubic(0.0, (double[2]) { rmin, zcurr }, psi_min, evcub->ctx);
    evcub->eval_cubic(0.0, (double[2]) { rup, zcurr }, psi_max, evcub->ctx);

    for (int ipsi=0; ipsi<nc[NPSI]; ++ipsi) {
      double psic_curr = psic_lo + ipsi*dpsi;
      pctx.Z = zcurr;

      // we continue to do root-finding for psi and not sqrt(psi)
      double psi_curr = psic_curr;
      if (inp->fl_coord == GKYL_MIRROR_GEOM_SQRT_PSI_CART_Z)
        psi_curr = psic_curr*psic_curr;
        
      pctx.psi =  psi_curr;

      struct gkyl_qr_res root = gkyl_ridders(psirz, &pctx, rmin, rup,
          psi_min[0]-psi_curr, psi_max[0]-psi_curr,
          100, 1e-10);

      if (root.status) {
        status = false;
        goto cleanup;
      }

      int idx[2] = { ipsi, iz };
      double *rz = gkyl_array_fetch(geo->nodesrz, gkyl_range_idx(&node_rng, idx));
      rz[0] = root.res; rz[1] = zcurr;
    }
  }

  cleanup:

  if (true != status)
    fprintf(stderr, "gkyl_mirror_geom_inew failed to generate a grid\n");
  
  gkyl_dg_basis_ops_evalf_release(evcub);
  
  return geo;
}

void
gkyl_mirror_geom_release(struct gkyl_mirror_geom *geom)
{
  gkyl_array_release(geom->nodesrz);
  gkyl_free(geom);
}

// for testing: will remove later
int
main(void)
{
  double clower[] = { 2.0e-6, 0.0, -2.0 };
  double cupper[] = { 1.0e-3, 2*M_PI, 2.0 };
  int cells[] = { 10, 16, 64 };

  // computational grid
  struct gkyl_rect_grid comp_grid;
  gkyl_rect_grid_init(&comp_grid, 3, clower, cupper, cells);

  if (!gkyl_check_file_exists("wham_hires.geqdsk_psi.gkyl")) {
    fprintf(stderr, "Unable to find file %s!\n", "wham_hires.geqdsk_psi.gkyl");
    goto cleanup;
  }
  
  // read psi(R,Z) from file
  struct gkyl_rect_grid psi_grid;
  struct gkyl_array *psi = gkyl_grid_array_new_from_file(&psi_grid, "wham_hires.geqdsk_psi.gkyl");

  // create mirror geometry
  struct gkyl_mirror_geom *geom =
    gkyl_mirror_geom_inew(&(struct gkyl_mirror_geom_inp) {
        .comp_grid = &comp_grid,
        
        .R = { psi_grid.lower[0], psi_grid.upper[0] },
        .Z = { psi_grid.lower[1], psi_grid.upper[1] },
        
        // psi(R,Z) grid size
        .nrnodes = psi_grid.cells[0]-1, // cells and not nodes
        .nznodes = psi_grid.cells[1]-1, // cells and not nodes

        .psiRZ = psi,
        .fl_coord = GKYL_MIRROR_GEOM_SQRT_PSI_CART_Z,
        .write_psi_cubic = true,
      }
    );

  // write out node coordinates
  struct gkyl_rect_grid nodal_grid;
  gkyl_rect_grid_init(&nodal_grid, 2,
    (double[]) { 0.0, 0.0 }, // arbitrary
    (double[]) { 1.0, 1.0 }, // arbitrary
    (int[]) { cells[0]+1, cells[2]+1 }
  );

  struct gkyl_range node_range;
  gkyl_range_init_from_shape(&node_range, 2, (int[2]) { cells[0]+1, cells[2]+1 });

  enum gkyl_array_rio_status status =
    gkyl_grid_sub_array_write(&nodal_grid, &node_range, 0, geom->nodesrz, "nodal_coords.gkyl");
  

  gkyl_mirror_geom_release(geom);
  gkyl_array_release(psi);

  cleanup:
  
  return 0;
}
