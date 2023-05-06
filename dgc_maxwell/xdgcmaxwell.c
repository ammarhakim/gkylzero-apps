#include <gkylzero.h>
#include <dgc_maxwell.h>

#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>

#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <stdint.h>
#include <stdbool.h>

bool sl_check_file_exists(const char *fname);
int64_t sl_file_size(const char *fname);

bool
sl_check_file_exists(const char *fname)
{
  return access(fname, F_OK) == 0;
}

int64_t
sl_file_size(const char *fname)
{
  struct stat st;
  stat(fname, &st);
  return st.st_size;
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

// free using gkyl_free
static char*
load_file(const char *fname)
{
  int64_t sz = sl_file_size(fname);
  char *buff = gkyl_malloc(sz+1);
  FILE *fp = fopen(fname, "r");
  int n = fread(buff, sz, 1, fp);
  fclose(fp);
  return buff;
}


int
main(int argc, char **argv)
{
  lua_State *L = luaL_newstate();
  lua_gc(L, LUA_GCSTOP, 0);
  luaL_openlibs(L);
  lua_gc(L, LUA_GCRESTART, -1);  
  
  if (argc > 1) {
    if (sl_check_file_exists(argv[1])) {
      char *buff = load_file(argv[1]);

      // set input file name as a global
      char set_inp_name[2000];
      snprintf(set_inp_name, sizeof set_inp_name, "DGC_INP_NAME = \"%s\"", argv[1]);
      run_lua(L, set_inp_name);
      
      run_lua(L, buff);
      gkyl_free(buff);
    }
  }
  lua_close(L);
  return 0;
}
