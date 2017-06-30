/*
 *   readIntFile.c
 *     Reads a comma separated list of integers from file.
 *       Assume list is small enough to fit in memory
 *       */


#include "randomInts.h"
#include <sys/time.h>

int getNextInt(FILE* fp)
{
  char c;
  char newnum[NDIGITS+1];
  for(int i=0; i<=NDIGITS; i++)
	newnum[i] = 0;
  int i=0;
  while( (c=fgetc(fp))!=',' && c!=EOF )
    newnum[i++] = c;
  return atoi(newnum);
}



randArray readIntsFromFile(const char* filename)
{
  FILE* fp = fopen(filename, "r");
  randArray array = {0,0};
  if(fp==NULL)
  {
    fprintf(stderr,"ERROR opening file %s\n", filename);
    return array;
  }

  // get number of values
  int bufSize = 20;
  char n_str[bufSize]; 
  fgets(n_str, bufSize, fp);
  int n = atoi(n_str);

  int *A = (int*)malloc(n*sizeof(int));

  for(int i=0; i<n; i++)
    A[i] = getNextInt(fp);
  fclose(fp);

  array.n = n;
  array.A = A;

  return array;
}



void writeRandomFile(int n, const char* filename)
{
	struct timeval t;
	gettimeofday(&t, NULL);	
    srand(t.tv_usec);

    int nMax = (int)pow(10,NDIGITS);
    FILE* fp = fopen(filename,"w");
    fprintf(fp,"%d\n",n);
    for(int i=0; i<n-1; i++)
        fprintf(fp,"%d, ",rand()%nMax);
    fprintf(fp,"%d",rand()%nMax);
    fclose(fp);
}
