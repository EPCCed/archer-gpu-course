MF=	Makefile

CC=	nvc
CFLAGS=  -O3 -ta=nvidia -Minfo=all 
#CC= gcc
#CFLAGS= -g

LFLAGS=	-lm

EXE=	reconstruct

SRC= \
	reconstruct.c \
	cio.c

#
# No need to edit below this line
#

.SUFFIXES:
.SUFFIXES: .c .o

OBJ=	$(SRC:.c=.o)

.c.o:
	$(CC) $(CFLAGS) -c $<

all:	$(EXE)

$(EXE):	$(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LFLAGS)

$(OBJ):	$(MF)

clean:
	rm -f $(OBJ) $(EXE) core
