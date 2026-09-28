# OptiX renderer for the 3D view. Needs the CUDA toolkit (nvcc) and an NVIDIA
# driver with OptiX (R535+); the OptiX headers are vendored in third_party/.
#
#   make -C flowscope/optix
#
# Produces build/libflowscope_optix.so and build/device.optixir, which
# flowscope/optix_backend.py loads at runtime.

NVCC ?= nvcc
OPTIX_INCLUDE ?= third_party/optix/include
ARCH ?= compute_75
BUILD := build

all: $(BUILD)/libflowscope_optix.so $(BUILD)/device.optixir

$(BUILD):
	mkdir -p $@

$(BUILD)/libflowscope_optix.so: src/host.cu src/shared.h | $(BUILD)
	$(NVCC) -O3 -std=c++17 -shared -Xcompiler -fPIC,-fvisibility=hidden \
		-I$(OPTIX_INCLUDE) -o $@ src/host.cu -ldl

$(BUILD)/device.optixir: src/device.cu src/shared.h src/vec.h | $(BUILD)
	$(NVCC) -O3 -std=c++17 --optix-ir -arch=$(ARCH) --use_fast_math -lineinfo -diag-suppress 20044 \
		-I$(OPTIX_INCLUDE) -o $@ src/device.cu

clean:
	rm -rf $(BUILD)

.PHONY: all clean
