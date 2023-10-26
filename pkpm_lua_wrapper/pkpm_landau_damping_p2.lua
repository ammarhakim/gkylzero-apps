local Plasma = G0.PKPM

local epsilon0 = 1.0
local mu0 = 1.0
local elcMass = 1.0
local elcCharge = -1.0
local elcTemp = 1.0
local n0 = 1.0
local knumber = 0.5
local perturb = 0.1

-- Maxwellian in 1x1v
local function maxwellian(n, vx, mass, temp)
   local v2 = vx*vx
   return n/math.sqrt(2*math.pi*temp/mass)*math.exp(-mass*v2/(2*temp))
end


pkpmApp = Plasma.App.new {
   tEnd = 20.0,
   nFrame = 1,

   lower = {-math.pi/knumber}, -- Configuration space lower left.
   upper = { math.pi/knumber}, -- Configuration space upper right.
   cells = {32}, -- Configuration space cells.
   basis = "serendipity", 
   polyOrder = 2, -- Polynomial order.
   timeStepper = "rk3", 

   -- Decomposition for configuration space.
   decompCuts = {1}, -- cuts in each configuration direction.

   -- Boundary conditions for configuration space.
   periodicDirs = {1}, -- Periodic directions.

   -- Electrons.
   elc = Plasma.Species.new {
      charge = elcCharge,  mass = elcMass,
      -- Velocity space grid.
      lower = {-6.0},
      upper = { 6.0},
      cells = {32},
      -- Initial conditions for distribution functions
      init_dist = function (t, xn)
         local x, v = xn[1], xn[2]
         local alpha = perturb
         local k = knumber
         local fv = (1 + alpha*math.cos(k*x))*maxwellian(n0, v, elcMass, elcTemp)
         local gv = elcTemp/elcMass*fv

         return fv, gv
      end,
      -- Initial conditions for momentum 
      init_fluid = function (t, xn)
         local x = xn[1]
         -- no initial flows
         return 0.0, 0.0, 0.0
      end,
      collisions = Plasma.Collisions.new {
         self_nu = function (t, xn)
            local x = xn[1]
            return 0.0
         end, 
      },
      evolve = true, -- Evolve species?
   },

   field = Plasma.Field.new {
      epsilon0 = epsilon0,  mu0 = mu0,
      init = function (t, xn)
         local alpha = perturb
         local k = knumber
         return -alpha*math.sin(k*xn[1])/k, 0.0, 0.0, 1.0, 0.0, 0.0
      end,
      evolve = true, -- Evolve field?
   }
}
pkpmApp:run()
