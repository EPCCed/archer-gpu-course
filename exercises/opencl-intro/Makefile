# 
# Makefile for OpenCL compiled via nvcc
# Note that the kernel source does not appear here
# other than as a dependency.
#

MF= Makefile
CP = cp

SRC= intro.c
EXE= intro

CC = nvcc
CFLAGS = -O3
LIB_OPENCL = -lOpenCL

default: $(EXE)

.SUFFIXES:
.SUFFIXES: .c  .o

OBJ=	$(SRC:.c=.o)

.c.o:
	$(CC) $(CFLAGS) -dc -c $<

$(EXE):	$(OBJ) kernel.c
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LIB_OPENCL) 

$(OBJ):	$(MF) 

clean:
	rm -f $(OBJ) $(EXE) core
