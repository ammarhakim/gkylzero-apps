#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>

#include <gkylzero.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static char*
load_file(const char *fname)
{
  int64_t sz = gkyl_file_size(fname);
  char *buff = malloc(sz+1);
  FILE *fp = fopen(fname, "r");
  int n = fread(buff, sz, 1, fp);
  fclose(fp);
  return buff;
}

static int
run_lua(lua_State *L, const char *str)
{
  if (luaL_loadstring(L, str) || lua_pcall(L, 0, LUA_MULTRET, 0)) {
    const char* ret = lua_tostring(L, -1);
    fprintf(stderr, "*** ERROR: %s\n", ret);
    return 1;
  }
  return 0;
}

int
main(int argc, char **argv)
{
  lua_State *L = luaL_newstate();
  lua_gc(L, LUA_GCSTOP, 0);
  luaL_openlibs(L);
  lua_gc(L, LUA_GCRESTART, -1);

  if (argc > 1) {
    if (gkyl_check_file_exists(argv[1])) {
      char *buff = load_file(argv[1]);
      run_lua(L, buff);
      free(buff);
    }
  }  

  lua_close(L);  
  return 0;
}
