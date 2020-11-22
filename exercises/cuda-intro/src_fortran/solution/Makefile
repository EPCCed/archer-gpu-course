# 
# Makefile cuda-intro Frrotran
#

MF=  Makefile
EXE= intro
SRC= array_negate.cuf  gpu_utils.cuf  intro.cuf  

ARCH=cc70

CC=	nvfortran
CFLAGS= -Mcuda=$(ARCH) -O3 

default: $(EXE)


.SUFFIXES:
.SUFFIXES: .cuf  .o

OBJ=	$(SRC:.cuf=.o)


.cuf.o:
	$(CC) $(CFLAGS) -c $<

$(EXE):	$(OBJ) 
	$(CC) $(CFLAGS) -o $@ $(OBJ) 

$(OBJ):	$(MF) 

clean:
	rm -f $(OBJ) $(EXE) *.mod core
