CUDA_HOME   = /Soft/cuda/6.5.14

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -I$(CUDA_HOME)/include -arch=sm_20 -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc 
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib

MARKETPLACE_EXE = Marketplace-Knapsack.exe
MARKETPLACE_OBJ = Marketplace-Knapsack.o

MARKETPLACE_WARPS_EXE = Marketplace-Knapsack-Warps.exe
MARKETPLACE_WARPS_OBJ = Marketplace-Knapsack-Warps.o

MARKETPLACE_THREADS_EXE = Marketplace-Knapsack-Threads.exe
MARKETPLACE_THREADS_OBJ = Marketplace-Knapsack-Threads.o

default: $(MARKETPLACE_EXE)

Marketplace-Knapsack.o: main.cu
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

$(MARKETPLACE_EXE): $(MARKETPLACE_OBJ)
	$(NVCC) $(MARKETPLACE_OBJ) -o $(MARKETPLACE_EXE) $(LD_FLAGS)

Marketplace-Knapsack-Warps.o: mainWarpsOptimized.cu
	$(NVCC) -c -o $@ mainWarpsOptimized.cu $(NVCC_FLAGS)

$(MARKETPLACE_WARPS_EXE): $(MARKETPLACE_WARPS_OBJ)
	$(NVCC) $(MARKETPLACE_WARPS_OBJ) -o $(MARKETPLACE_WARPS_EXE) $(LD_FLAGS)

Marketplace-Knapsack-Threads.o: mainWorkPerThreadOptimized.cu
	$(NVCC) -c -o $@ mainWorkPerThreadOptimized.cu $(NVCC_FLAGS)

$(MARKETPLACE_THREADS_EXE): $(MARKETPLACE_THREADS_OBJ)
	$(NVCC) $(MARKETPLACE_THREADS_OBJ) -o $(MARKETPLACE_THREADS_EXE) $(LD_FLAGS)

all:	$(MARKETPLACE_EXE) $(MARKETPLACE_WARPS_EXE) $(MARKETPLACE_THREADS_EXE)

clean:
	rm -rf *.o Marketplace-Knapsack*.exe

clcu:
	rm -rf Marketplace-Knapsack*.o* Marketplace-Knapsack*.e*

