# 
# Makefile for reverse.cu exercise
#

MF= Makefile
CP= cp

EXE= reverse

SRC= reverse.cu

ARCH=sm_35

CC=	nvcc
CFLAGS= -O3 -arch=$(ARCH)

.SUFFIXES:
.SUFFIXES: .cu  .o

OBJ=	$(SRC:.cu=.o)


.cu.o:
	$(CC) $(CFLAGS) -dc -c $<

default: $(EXE)

$(EXE):	$(OBJ) 
	$(CC) $(CFLAGS) -o $@ $(OBJ) 

$(OBJ):	$(SRC) $(MF) 

clean:
	rm -f $(OBJ) $(EXE) core
