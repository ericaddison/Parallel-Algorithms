//shared_scalar.upc
#include <stdio.h>
#include <upc.h>

shared int w = 0;			// shared scalar
upc_lock_t* lock = upc_all_lock_alloc();
int main() {

	upc_lock(lock); 		/* Get lock */
	w += 1;
	upc_unlock(lock); 		/* Release lock */
	
	upc_barrier;
	if (MYTHREAD == 0)	
	{
		upc_lock_free(lock); /* Free */
		printf("%d\n", w);
	}
    
    return 0;
}
