FFTW_LDLIBS = -lfftw3
FFTW_INCLUDES=
FFTW_LDFLAGS=
COMMON_FLAGS = -Wall -pedantic -O3
CFLAGS = $(COMMON_FLAGS) $(FFTW_INCLUDES) -std=c99 -fPIC -Iinclude

default: lib/libresample.so bin/benchmark

bin/benchmark: lib/libresample.so include/timer.h include/interpolate.h

lib/%.o: lib/%.c
	$(CC) $(CFLAGS) -c $< -o $@

lib/libresample.so: lib/interpolate.o lib/timer.o lib/phase_shift_interface.o lib/allocation.o
	$(CC) -shared -fPIC $(FFTW_LDFLAGS) $^ -o $@ $(FFTW_LDLIBS)

bin/%: bin/%.c
	$(CC) $(CFLAGS) $< -Llib -lresample -Wl,-rpath=../lib -o $@

lib/timer.o: include/timer.h
lib/interpolate.o: include/timer.h include/interpolate.h

clean: 
	rm -f lib/libresample.so lib/interpolate.o lib/phase_shift_interface.o \
	  lib/allocation.o lib/timer.o bin/benchmark

.PHONY: clean
