interpolate: interpolate.h

CFLAGS = -std=c99 -O2 -Wall
LIBADD = -lfftw3

interpolate: interpolate.o
	$(CC) $(LDFLAGS) $< $(LIBADD) -o $@

interpolate.o: interpolate.h

clean: 
	rm -f interpolate interpolate.o

.PHONY: clean
