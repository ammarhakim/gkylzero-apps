# -*- makefile-gmake -*-

# Sample Makefile to use installed gkylzero library: copy and modify
# for your needs

BUILD_DIR ?= build

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

CFLAGS = -O3 -g -ffast-math -I.

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

# On OSX we should use Accelerate framework
ifeq ($(UNAME), Darwin)
	LAPACK_INC = . # dummy
	LAPACK_LIB = -framework Accelerate
	CFLAGS += -DGKYL_USING_FRAMEWORK_ACCELERATE
endif

# Read MPI paths and flags if needed 
USING_MPI =
MPI_INC_DIR = . # dummy
MPI_LIB_DIR = .
ifeq (${USE_MPI}, 1)
	USING_MPI = yes
	MPI_INC_DIR = ${MPI_INC}
	MPI_LIB_DIR = ${MPI_LIB}
	MPI_LIBS = -lmpi
	CFLAGS += -DGKYL_HAVE_MPI
endif

INCLUDES = -I${G0_INC_DIR} -I${LAPACK_INC} -I${SUPERLU_INC} -I${MPI_INC_DIR} -Iinp
LIB_DIRS = -L${LAPACK_LIB_DIR} -L${SUPERLU_LIB_DIR} -L${MPI_LIB_DIR}
EXT_LIBS = ${LAPACK_LIB} ${SUPERLU_LIB} ${MPI_LIBS}

# List of executables to build
REGS := $(patsubst %.c,${BUILD_DIR}/%,$(wildcard inp/rt_*.c))

all: ${REGS} ## Build inp tests

# Inp tests
${BUILD_DIR}/inp/%: inp/%.c
	$(MKDIR_P) ${BUILD_DIR}/inp
	${CC} ${CFLAGS} ${LDFLAGS} -o $@ $< -I. $(INCLUDES) -L${G0_LIB_DIR} ${G0_RPATH} ${G0_LIBS} ${LIB_DIRS} ${EXT_LIBS}

# command to make dir
MKDIR_P ?= mkdir -p

clean:
	rm -rf build

