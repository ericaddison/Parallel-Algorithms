/*
 randomFileGenerator.c
   Outputs a file containing a list of comma separated integers
   in the range [0-999]. Number of values in the file is given by command
   line argument #1, and file name by optional argument #2
*/

#include "randomInts.h"
#define USAGE "Usage: randomFileGenerator Nvalues [filename]\n"

int main(int argc, char** argv)
{

    if(argc==1)
    {
    	printf(USAGE);
        return 1;
    }

// convert arg1 to int
    int n = atoi(argv[1]);
    if(n<=0)
        return 0;

    char filename[256];
    if(argc>=3)
        strncpy(filename,argv[2],256);
    else
        sprintf(filename,"randomInts_%d.dat",n);

	writeRandomFile(n, filename);

    return 0;
}
