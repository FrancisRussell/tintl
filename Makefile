interpolate: interpolate.h

CFLAGS = -std=c99 -O2 -Wall -pedantic
LIBADD = -lfftw3

interpolate: interpolate.o timer.o
	$(CC) $(LDFLAGS) $^ $(LIBADD) -o $@

interpolate.o: interpolate.h timer.h

timer.o: timer.h

clean: 
	rm -f interpolate interpolate.o timer.o

.PHONY: clean
