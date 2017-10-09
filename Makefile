
CC := gcc
CFLAGS += -fPIC -std=c99 -Wall -lm -Wextra 
CFLAGS += -O3

all: clike.so clike-parallel.so cmuselike.so cmuselike-parallel.so clustering

clustering: 
	$(MAKE) -C clustering/

%-parallel.so: %.c
	${CC} ${CFLAGS} -fopenmp -DPARALLEL=1 $< -o $@ -shared

%.so: %.c
	${CC} ${CFLAGS} $< -o $@ -shared
clean: 
	rm *.so

.PHONY: all clean clustering

