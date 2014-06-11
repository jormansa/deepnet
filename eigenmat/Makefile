INCLUDE := -I/opt/local/include/eigen3/ -I/usr/local/include/eigen3/ -I$(HOME)/eigen
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	CCFLAGS := -O3 -fPIC -fopenmp
endif
ifeq ($(UNAME_S),Darwin)
	CCFLAGS := -O3 -fPIC -arch i386 -arch x86_64
endif
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Linux)
	CCFLAGS += -shared
	LIB_NAME := libeigenmat.so
else
ifeq ($(UNAME_S), Darwin)
	CCFLAGS += -dynamiclib
	LIB_NAME := libeigenmat.dylib
else
	CCFLAGS +=  -shared
	LIB_NAME := libeigenmat.dll
endif
endif
all: eigenmat.cc eigenmat.h ziggurat.cc ziggurat.h
	g++ $(CCFLAGS) $(INCLUDE) -o $(LIB_NAME) eigenmat.cc ziggurat.cc

clean:
	rm -rf *.linkinfo *.pyc *.so
