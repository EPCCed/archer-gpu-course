MF=	Makefile

FC=pgf90
FFLAGS=-O3 -ta=nvidia -Minfo=accel 

LFLAGS=	

EXE=	reconstruct

SRC= \
	reconstruct.f90 \
	fio.f90

#
# No need to edit below this line
#

.SUFFIXES:
.SUFFIXES: .f90 .o

OBJ=	$(SRC:.f90=.o)

.f90.o:
	$(FC) $(FFLAGS) -c $<

all:	$(EXE)

$(EXE):	$(OBJ)
	$(FC) $(FFLAGS) -o $@ $(OBJ) $(LFLAGS)

$(OBJ):	$(MF)

clean:
	rm -f $(OBJ) $(EXE) core
