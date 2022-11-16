include ../Makefile.inc

SRC_CU          = $(shell find ./ -name '*.cu')

all: LayerNormPlugin.so

%.o: %.cu
	$(NVCC) $(CUFLAG) $(INCLUDE) -o $@ -c $<

LayerNormPlugin.so: $(SRC_CU:.cu=.o)
	$(NVCC) $(SOFLAG) $(LDFLAG) -o $@ $^

.PHONY: clean
clean:
	rm -rf ./*.so ./*.o ./*.d ./*.trt ./temp* ./*.nsys-rep ./*.sqlite

.PHONY: test
test:
	clear
	make clean
	make all
	nsys nvprof -o temp --print-gpu-trace python testLayerNormPluginOnnx.py

