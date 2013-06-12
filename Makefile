FFTW_LDLIBS = -lfftw3
FFTW_INCLUDES = -I/usr/include
FFTW_LDFLAGS=
COMMON_FLAGS = -Wall -pedantic -O3 -fPIC $(FFTW_INCLUDES) -Iinclude
CFLAGS = $(COMMON_FLAGS) -std=c99

default: lib/libresample.so bin/benchmark

bin/benchmark.o: include/timer.h include/interpolate.h

lib/libresample.so: lib/interpolate.o lib/timer.o lib/phase_shift_interface.o lib/allocation.o
	$(CC) -shared -fPIC $(FFTW_LDFLAGS) $^ -o $@ $(FFTW_LDLIBS)

bin/%: bin/%.o lib/libresample.so
	$(CC) $< -o $@ -Llib -lresample -Wl,-rpath=../lib -Wl,-rpath=./lib

lib/timer.o: include/timer.h

lib/interpolate.o: include/timer.h include/interpolate.h

clean: 
	rm -f lib/libresample.so lib/interpolate.o lib/phase_shift_interface.o \
	  lib/allocation.o lib/timer.o bin/benchmark.o bin/benchmark

.PHONY: default clean
