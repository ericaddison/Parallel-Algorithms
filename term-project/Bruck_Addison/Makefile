all: 
	source env.sh; \
	make -C cpp; \
	make -C cuda
	module load intel; \
	make -C cilk

clean:
	make -C cpp clean
	make -C cuda clean
	make -C cilk clean
