include ./Makefile.inc

export MAKEROOT := $(shell pwd)

SOURCE_CPP  = $(shell find . -name '*.cpp' 2>/dev/null)
SOURCE_PY   = $(shell find . -name '*.py' 2>/dev/null)
OBJ         = $(shell find . -name *.o 2>/dev/null)
DEP         = $(OBJ:.o=.d)
TARGET_EXE  = $(SOURCE_CPP:.cpp=.exe)
OUT_BIN		= $(MAKEROOT)/../bin
OUT_SO		= $(MAKEROOT)/../so

-include $(DEP)

all: $(TARGET_EXE)

%.exe: %.o
	$(NVCC) $(CCFLAG) $(LDFLAG) -o $(OUT_BIN)/$@ $+

%.o: %.cpp
	$(NVCC) $(CCFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAG) $(INCLUDE) -Xcompiler -fPIC -o $@ -c $<

.PHONY: test
test:
	make clean
	make
	$(OUT_BIN)/$(TARGET_EXE) perf
	python utils/local_evaluate.py ../perf.res.txt
	$(OUT_BIN)/$(TARGET_EXE) label
	python utils/local_evaluate.py ../label.res.txt

.PHONY: clean
clean:
	rm -rf ./*.d ./*.o $(OUT_SO)/*.so $(OUT_BIN)/*.exe ./*.plan