CC     = nvcc
# BENCH: sm_86 = Ampere GA10x (RTX 3060 Ti, RTX A2000); -lineinfo lets ncu map metrics to source lines
FLAGS  = -O3 -arch=sm_86 -lineinfo
TARGET = scan.out
SRC    = main.cu
DEPS   = ../bench.cuh ../common.cuh
# BENCH: smaller default than the other kernels; naive is O(N^2)
N     ?= 2^16

.PHONY: all run clean

all: $(TARGET)

# BENCH: the shared headers are dependencies so editing them triggers a rebuild
$(TARGET): $(SRC) $(DEPS)
	$(CC) $(FLAGS) $(SRC) -o $(TARGET)

# BENCH: make run            -> N = 2^16
#        make run N=2^20     -> any size
run: $(TARGET)
	./$(TARGET) $(N)

clean:
	rm -f *.out
