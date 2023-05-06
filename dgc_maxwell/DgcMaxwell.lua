-- module for running DGC Maxwell via Lua
local ffi = require "ffi"

ffi.cdef [[

// Update status
struct gkyl_update_status {
  bool success; // status of update
  double dt_actual; // actual time-step taken
  double dt_suggested; // suggested stable time-step
};

/**
 * Type of function to project.
 *
 * @param t Time to evaluate function
 * @param xn Coordinates for evaluation
 * @param fout Output vector of 'num_ret_vals'
 * @param ctx Context for function evaluation. Can be NULL
 */
typedef void (*evalf_t)(double t, const double *xn, double *fout, void *ctx);

typedef struct dgc_app dgc_app;

// App container
struct dgc_app_cont {
  double tEnd;
  int nFrame;

  struct dgc_app *app;
};

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
]]

-- gkyl_util.h
ffi.cdef [[
/**
 * Time-trigger. Typical initialization is:
 * 
 * struct gkyl_tm_trigger tmt = { .dt = tend/nframe };
 */
struct gkyl_tm_trigger {
  int curr; // current counter
  double dt, tcurr; // Time-interval, current time
};

/**
 * Check if the tcurr should trigger and bump internal counters if it
 * does. This only works if sequential calls to this method have the
 * tcurr monotonically increasing.
 *
 * @param tmt Time trigger object
 * @param tcurr Current time.
 * @return 1 if triggered, 0 otherwise
 */
int gkyl_tm_trigger_check_and_bump(struct gkyl_tm_trigger *tmt, double tcurr);

/**
 * Compute time in seconds since epoch.
 *
 * @return Time in seconds
 */
double gkyl_time_now(void);
]]

local _M = { }

-- Time in seconds from epoch
_M.time_now = function()
   return ffi.C.gkyl_time_now()
end

-- time-trigger object
local tm_trigger_type = ffi.typeof("struct gkyl_tm_trigger")
local tm_trigger_mt = {
   __new = function(self, dt)
      local tmt = ffi.new(self, { curr = 0, dt = dt, tcurr = 0.0 })
      return tmt
   end,
   __index = {
      checkAndBump = function(self, tcurr)
	 return ffi.C.gkyl_tm_trigger_check_and_bump(self, tcurr) == 1
      end
   }
}
_M.TimeTrigger = ffi.metatype(tm_trigger_type, tm_trigger_mt)

-- Wraps user given function in a function that can be passed to the C
-- callback APIs
local function eval_cfunc(func)
   return function(t, xn, fout, ctx)
      local xnl = ffi.new("double[10]")
      for i=1,3 do xnl[i] = xn[i-1] end
      local ret = { func(t, xnl) } -- package return into table
      for i=1,#ret do
         fout[i-1] = ret[i]
      end
   end
end

-- App
local app_type = ffi.typeof("struct dgc_app_cont")
local app_mt = {
   __new = function(self, tbl)
      local inp = ffi.new("struct dgc_inp")

      local s, e = string.find(DGC_INP_NAME, ".lua")
      local name = string.sub(DGC_INP_NAME, 1, s-1)

      inp.name = name
      
      inp.ndim = #tbl.cells
      for d=1, inp.ndim do
         inp.lower[d-1] = tbl.lower[d]
         inp.upper[d-1] = tbl.upper[d]
         inp.cells[d-1] = tbl.cells[d]
      end

      inp.cfl_frac = 0.95
      if tbl.cflFrac then
         inp.cfl_frac = tbl.cflFrac
      end

      inp.init_E = eval_cfunc(tbl.init_E)
      inp.init_B = eval_cfunc(tbl.init_B)

      local app_cont = ffi.new(self)
      
      app_cont.tEnd = tbl.tEnd
      app_cont.nFrame = tbl.nFrame
      app_cont.app = ffi.C.dgc_app_new(inp)

      return app_cont
   end,
   __gc = function(self)
      ffi.C.dgc_app_release(self.app)
   end,
   __index = {
      init = function(self)
         ffi.C.dgc_app_apply_ics(self.app)
      end,
      write = function(self, tm, frame)
         ffi.C.dgc_app_write(self.app, tm, frame)
      end,
      writeFieldEnergy = function(self)
	 ffi.C.dgc_app_write_em_energy(self.app)
      end,      
      update = function(self, dt)
         return ffi.C.dgc_app_update(self.app, dt)
      end,
      calcFieldEnergy = function(self, tcurr)
	 return ffi.C.dgc_app_calc_em_energy(self.app, tcurr)
      end,      
      run = function(self)
	 local frame_trig = _M.TimeTrigger(self.tEnd/self.nFrame)

	 -- function to write data to file
	 local function writeData(tcurr)
	    if frame_trig:checkAndBump(tcurr) then
	       self:write(tcurr, frame_trig.curr-1)
	    end
	 end

	 local p1_trig = _M.TimeTrigger(self.tEnd/10)
	 -- log messages
	 local function writeLogMessage(tcurr, step, dt)
	    if p1_trig:checkAndBump(tcurr) then
	       io.write(string.format(" Step %6d %.4e. Time-step  %.6e \n", step, tcurr, dt))
	    end
	 end

	 io.write(string.format("Starting GkeyllZero simulation\n"))
	 io.write(string.format("  tstart: %.6e. tend: %.6e\n", 0.0, self.tEnd))

	 local tinit0 = _M.time_now()
	 self:init()
	 io.write(string.format("  Initialization completed in %g sec\n", _M.time_now() - tinit0))
	 
	 self:calcFieldEnergy(0.0)
	 writeData(0.0)

	 local tloop0 = _M.time_now()
	 local tcurr, tend = 0.0, self.tEnd
	 local dt =  tend-tcurr
	 local step = 1
	 while tcurr < tend do
	    local status = self:update(dt)
	    tcurr = tcurr + status.dt_actual

	    if status.success == false then
	       io.write(string.format("***** Simulation failed at step %5d at time %.6e\n", step, tcurr))
	       break
	    end

	    self:calcFieldEnergy(tcurr)

	    writeLogMessage(tcurr, step, status.dt_actual)
	    writeData(tcurr)

	    dt = math.min(status.dt_suggested, (tend-tcurr)*(1+1e-6))
	    step = step + 1
	 end

	 self:writeFieldEnergy()

	 local tloop1 = _M.time_now()
	 
	 io.write(string.format("Completed in %d steps (tend: %.6e). \n", step-1, tcurr))
	 io.write(string.format("Main loop took %g secs to complete\n", tloop1-tloop0))
      end,      
   }
}
_M.App = ffi.metatype(app_type, app_mt)

return _M
