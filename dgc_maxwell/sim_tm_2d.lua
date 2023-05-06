local DgcMaxwell = require "DgcMaxwell"

-- sim parameters
local LX = 8.0
local LY = 4.0
local m = 8
local n = 4

local app = DgcMaxwell.App {
   tEnd = 10,
   nFrame = 1,

   lower = { 0.0, 0.0 },
   upper = { LX, LY },
   cells = { 80, 40 },
   cflFrac = 1.0,

   init_E = function (t, xn)
      local x, y = xn[1], xn[2]
      local Ex, Ey = 0.0, 0.0
      local a = m*math.pi/LX
      local b = n*math.pi/LY
      return 0.0, 0.0, math.sin(a*x)*math.sin(b*y)
   end,

   init_B = function (t, xn)
      return 0.0, 0.0, 0.0
   end
}
app:run()
