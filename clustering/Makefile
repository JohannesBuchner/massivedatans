
CC := gcc
CFLAGS += -fPIC -std=c99 -Wall -lm -Wextra 
CFLAGS += -O3

all: cneighbors.so cneighbors-parallel.so cclustering.so cclustering-parallel.so

%-parallel.so: %.c
	${CC} ${CFLAGS} -fopenmp -DPARALLEL=1 $< -o $@ -shared

%.so: %.c
	${CC} ${CFLAGS} $< -o $@ -shared
clean: 
	rm *.so

.PHONY: all clean

