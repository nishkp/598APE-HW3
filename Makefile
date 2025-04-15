FUNC := g++
copt := -c 
OBJ_DIR := ./bin/
FLAGS := -Ofast -fopenmp -lm -g -Werror
# using fast (which implements -Ofast doesnt seem to affect location)

# -fno-inline -fno-default-inline

CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix $(OBJ_DIR),$(notdir $(CPP_FILES:.cpp=.obj)))

all:
	$(FUNC) ./main.cpp -o ./main.exe $(FLAGS)

clean:
	rm -f ./*.exe