NVCC ?= nvcc
ARCH ?= native
N ?= 1048579

TARGET := vector_add
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo

.PHONY: all build run sanitize clean

all: build

build:
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) vector_add.cu -o $(TARGET)

run: build
	./$(TARGET) $(N)

sanitize: build
	compute-sanitizer --tool memcheck --error-exitcode=2 ./$(TARGET) 257

clean:
	rm -f $(TARGET)
