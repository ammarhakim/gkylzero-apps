#include <stdio.h>

#ifdef GKYL_HAVE_LUA

#include <dgc_maxwell.h>

#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>

#include <lautoc.h>

int add_two_nums(int a, int b)
{
  return a + b;
}
luaA_function_declare(add_two_nums, int, int, int);
    
static int
add_two_nums_lua(lua_State *L)
{
  luaA_call(L, add_two_nums);
  return 1;
}

static const luaL_Reg dgc_lib[] = {
  {"add_two_nums", add_two_nums_lua},
  {NULL, NULL}
};

// Top-level Lua wrapper around DGC Maxwell app
struct dgc_maxwell_lua_app {
  lua_State *L;
};


struct dgc_maxwell_lua_app *
dgc_maxwell_lua_app_new(void)
{
  struct dgc_maxwell_lua_app *app = gkyl_malloc(sizeof(*app));

  app->L = luaL_newstate();
  lua_gc(app->L, LUA_GCSTOP, 0);
  luaL_openlibs(app->L);
  luaA_open(app->L);
  lua_gc(app->L, LUA_GCRESTART, -1);

  // register various functions
  luaA_function_register(app->L, add_two_nums, int, int, int);
  luaL_newlib(app->L, dgc_lib);
  
  return app;
}

void
dgc_maxwell_lua_app_release(struct dgc_maxwell_lua_app *app)
{
  lua_close(app->L);
  gkyl_free(app);
}

    
    
int
main(int argc, char **argv)
{
  struct dgc_maxwell_lua_app *app = dgc_maxwell_lua_app_new();


  dgc_maxwell_lua_app_release(app);
  
  return 0;
}

#else

int
main(int argc, char **argv)
{
  fprintf(stderr, "Not built with Lua!\n");
  return 0;
}

#endif
