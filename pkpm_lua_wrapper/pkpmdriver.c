#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>

#include <gkylzero.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int
calc_output_prefix_len(const char *fn)
{
  const char *suff = strrchr(fn, '.');
  return strlen(fn) - (suff ? strlen(suff) : 0);
}

int
main(int argc, char **argv)
{
  lua_State *L = luaL_newstate();
  lua_gc(L, LUA_GCSTOP, 0);
  luaL_openlibs(L);
  gkyl_pkpm_lw_openlibs(L);
  lua_gc(L, LUA_GCRESTART, -1);

  if (argc > 1) {
    if (gkyl_check_file_exists(argv[1])) {
      // set file prefix as a global
      lua_pushlstring(L, argv[1], calc_output_prefix_len(argv[1]));
      lua_setglobal(L, "GKYL_OUT_PREFIX");
      
      int64_t sz;
      char *buff = gkyl_load_file(argv[1], &sz);
      glua_run_lua(L, buff, sz, stderr);
      gkyl_free(buff);
    }
  }
  lua_close(L);  
  return 0;
}
