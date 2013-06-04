COMMON_FLAGS = -Wall -pedantic -O2
CFLAGS = ${COMMON_FLAGS} -std=c99 -fPIC -Iinclude
LIBADD = -lfftw3

default: lib/libresample.so bin/benchmark

bin/benchmark: lib/libresample.so include/timer.h include/interpolate.h

lib/%.o: lib/%.c
	${CC} ${CFLAGS} -c $< -o $@

lib/libresample.so: lib/interpolate.o lib/timer.o lib/phase_shift_interface.o lib/allocation.o
	$(CC) -shared -fPIC $^ -o $@ $(LIBADD)

bin/%: bin/%.c
	${CC} ${CFLAGS} $< -Llib -lresample -lfftw3 -Wl,-rpath=../lib -o $@

lib/timer.o: include/timer.h
lib/interpolate.o: include/timer.h include/interpolate.h

clean: 
	rm -f lib/libresample.so lib/interpolate.o lib/phase_shift_interface.o \
	  lib/allocation.o lib/timer.o bin/benchmark

.PHONY: clean
