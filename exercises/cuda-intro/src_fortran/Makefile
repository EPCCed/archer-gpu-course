# 
# Makefile cuda-intro Frrotran
#

MF=  Makefile
EXE= intro
SRC= array_negate.cuf  gpu_utils.cuf  intro.cuf  

CC=     nvfortran
CFLAGS= -Mcuda -O3 
# required for cublas linking
LIBS=${MATHLIB}

default: $(EXE)


.SUFFIXES:
.SUFFIXES: .cuf  .o

OBJ=	$(SRC:.cuf=.o)


.cuf.o:
	$(CC) $(CFLAGS) -c $<

$(EXE):	$(OBJ) 
	$(CC) $(CFLAGS) -o $@ $(OBJ) -L$(LIBS)

$(OBJ):	$(MF) 

clean:
	rm -f $(OBJ) $(EXE) *.mod core
