# 
# Makefile for reverse.cuf exercise
#

MF= Makefile
CP= cp

EXE= reverse

SRC= reverse_kernel.cuf gpu_utils.cuf reverse.cuf  

ARCH=cc70

CC=	nvfortran
CFLAGS= -Mcuda=$(ARCH) -O3 


.SUFFIXES:
.SUFFIXES: .cuf  .o

OBJ=	$(SRC:.cuf=.o)


.cuf.o:
	$(CC) $(CFLAGS) -c $<

default: $(EXE)

$(EXE):	$(OBJ) 
	$(CC) $(CFLAGS) -o $@ $(OBJ) 

$(OBJ):	$(MF) 

clean:
	rm -f $(OBJ) $(EXE) *.mod core
