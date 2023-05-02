# 
# Makefile cuda recontruct C version
#

MF= Makefile

EXE= reconstruct
SRC= reconstruct.cu reconstruct_kernels.cu utilities.cu

ARCH=sm_60
CC=	nvcc
CFLAGS= -O3 -arch=$(ARCH)

default: $(EXE)

.SUFFIXES:
.SUFFIXES: .cu  .o

OBJ=	$(SRC:.cu=.o)


.cu.o:
	$(CC) $(CFLAGS) -dc -c $<

$(EXE):	$(OBJ) 
	$(CC) $(CFLAGS) -o $@ $(OBJ) 

$(OBJ):	$(MF) *.h

clean:
	rm -f $(OBJ) $(EXE) core output.pgm
