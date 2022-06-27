# 
# Makefile cuda-intro Fortran

MF=  Makefile
EXE= intro
SRC= scale_vector.cuf  gpu_utils.cuf  intro.cuf  

CC=     nvfortran
CFLAGS= -Mcuda -O3 
# if required for cublas linking
LIBS= #-L${MATHLIB}

default: $(EXE)


.SUFFIXES:
.SUFFIXES: .cuf  .o

OBJ=	$(SRC:.cuf=.o)


.cuf.o:
	$(CC) $(CFLAGS) -c $<

$(EXE):	$(OBJ) 
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LIBS)

$(OBJ):	$(MF) 

clean:
	rm -f $(OBJ) $(EXE) *.mod core
