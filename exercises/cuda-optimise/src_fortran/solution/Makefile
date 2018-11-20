# 
#Makefile
#

MF= Makefile

EXE= reconstruct

SRC= reconstruct_kernels.cuf  reconstruct.cuf 


ARCH=cc20
ifeq ($(HOSTNAME),gpulab2.hydra)
ARCH=cc35
endif


CC=	pgfortran
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
