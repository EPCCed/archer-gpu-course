EXE_NAME = "04_Exercise"

default : $(EXE)

SRC = $(wildcard *.cpp)

CXXFLAGS = -O3
LINK = ${CXX}
LINKFLAGS =

DEPFLAGS = -M

OBJ = $(SRC:.cpp=.$(KOKKOS_DEVICES).o)
LIB =


$(EXE): $(OBJ)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: 
	rm -f *.o *.cuda *.host

# Compilation rules

%.$(KOKKOS_DEVICES).o:%.cpp
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $< -o $@

test: $(EXE)
	./$(EXE)
