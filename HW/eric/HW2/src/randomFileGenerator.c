/*
  randomFileGenerator.c
  Outputs a file containing a list of comma separated integers
  in the range [0-999]. Number of values in the file is given by command
  line argument #1, and file name by optional argument #2
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define USAGE "Usage: randomFileGenerator Nvalues [filename]\n"

int main(int argc, char** argv)
{

// check if no arguments
if(argc==1)
{
  printf(USAGE);
  return 1;
}

// convert arg1 to int
int n = atoi(argv[1]);
if(n<=0)
  return 0;

// check for filename argument
char filename[256];
if(argc>=3)
  strncpy(filename,argv[2],256);
else
  sprintf(filename,"randomInts_%d.dat",n);

// initialize random seed
time_t t;
srand((unsigned)time(&t));

// write to file
FILE* fp = fopen(filename,"w");
for(int i=0; i<n-1; i++)
  fprintf(fp,"%d, ",rand()%1000);
fprintf(fp,"%d",rand()%1000);
fclose(fp);

return 0;
}
