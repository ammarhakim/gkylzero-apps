# -*- makefile-gmake -*-

# Sample Makefile to use installed gkylzero library: copy and modify
# for your needs

# Default lapack include and libraries: we prefer linking to static library
LAPACK_INC = ${HOME}/gkylsoft/OpenBLAS/include
LAPACK_LIB_DIR = ${HOME}/gkylsoft/OpenBLAS/lib
LAPACK_LIB = -lopenblas

# SuperLU includes and librararies
SUPERLU_INC = ${HOME}/gkylsoft/superlu/include
ifeq ($(UNAME_S),Linux)
	SUPERLU_LIB_DIR = ${HOME}/gkylsoft/superlu/lib64
	SUPERLU_LIB = ${HOME}/gkylsoft/superlu/lib64/libsuperlu.a
else
	SUPERLU_LIB_DIR = ${HOME}/gkylsoft/superlu/lib
	SUPERLU_LIB = ${HOME}/gkylsoft/superlu/lib/libsuperlu.a
endif

PREFIX = ${HOME}/gkylsoft

# Include config.mak file (if it exists)
-include config.mak

CFLAGS = -O3 -g -ffast-math -I. -fPIC

G0_INC_DIR = ${PREFIX}/gkylzero/include
G0_LIB_DIR = ${PREFIX}/gkylzero/lib
G0_LIB = -lgkylzero
CUDA_LIB = 

ifeq ($(CC), nvcc)
	CUDA_LIB = -lcublas
endif

G0_LIBS = ${G0_LIB} ${CUDA_LIB} -lm -lpthread
G0_RPATH = -Wl,-rpath,${G0_LIB_DIR}

# determine OS we are running on
UNAME = $(shell uname)

EXTRA_LINK_FLAGS = 

# On OSX we should use Accelerate framework
ifeq ($(UNAME), Darwin)
	LAPACK_INC = . # dummy
	LAPACK_LIB = -framework Accelerate
	CFLAGS += -DGKYL_USING_FRAMEWORK_ACCELERATE
	EXTRA_LINK_FLAGS += 
endif

# On Linux we need to pass proper link flags to export all symbols
ifeq ($(UNAME), Linux)
	EXTRA_LINK_FLAGS += -Wl,-E
endif

# Read MPI paths and flags if needed 
USING_MPI =
MPI_INC_DIR = . # dummy
MPI_LIB_DIR = .
ifeq (${USE_MPI}, 1)
	USING_MPI = yes
	MPI_INC_DIR = ${CONF_MPI_INC_DIR}
	MPI_LIB_DIR = ${CONF_MPI_LIB_DIR}
	MPI_LIBS = -lmpi
	CFLAGS += -DGKYL_HAVE_MPI
endif

# Read LUA paths and flags if needed 
USING_LUA =
LUA_INC_DIR = . # dummy
LUA_LIB_DIR = .
ifeq (${USE_LUA}, 1)
	LUA_INC_DIR = ${CONF_LUA_INC_DIR}
	LUA_LIB_DIR = ${CONF_LUA_LIB_DIR}
	LUA_LIBS = -l${CONF_LUA_LIB}
	CFLAGS += -DGKYL_HAVE_LUA
endif

INCLUDES = -I${G0_INC_DIR} -I${LAPACK_INC} -I${SUPERLU_INC} -I${MPI_INC_DIR} -I${LUA_INC_DIR}
LIB_DIRS = -L${LAPACK_LIB_DIR} -L${SUPERLU_LIB_DIR} -L${MPI_LIB_DIR} -L${LUA_LIB_DIR}
EXT_LIBS = ${LAPACK_LIB} ${SUPERLU_LIB} ${MPI_LIBS} 

all: pkpmdriver

pkpmdriver: pkpmdriver.c
	${CC} ${CFLAGS} pkpmdriver.c -o pkpmdriver ${INCLUDES} ${EXTRA_LINK_FLAGS} -L${G0_LIB_DIR} ${G0_RPATH} ${G0_LIBS} ${LIB_DIRS} ${EXT_LIBS} ${LUA_LIBS} -ldl

.PHONY: clean
clean: ## Clean build outputs
	rm -rf pkpmdriver

# From: https://www.client9.com/self-documenting-makefiles/
.PHONY: help
help: ## Show help
	@echo "Following targets are defined:"
	@awk -F ':|##' '/^[^\t].+?:.*?##/ {\
        printf "\033[36m%-30s\033[0m %s\n", $$1, $$NF \
        }' $(MAKEFILE_LIST)
