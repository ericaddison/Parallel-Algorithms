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

void readIntsFromFile(const char* filename, int n, int* array)
{
  FILE* fp = fopen(filename, "r");
  if(fp==NULL)
  {
    fprintf(stderr,"ERROR opening file %s\n", filename);
    return;
  }

  for(int i=0; i<n; i++)
    array[i] = getNextInt(fp);
  fclose(fp);
}


void writeRandomFile(int n, const char* filename)
{
	struct timeval t;
	gettimeofday(&t, NULL);	
    srand(t.tv_usec);

    int nMax = (int)pow(10,NDIGITS);
    FILE* fp = fopen(filename,"w");
    for(int i=0; i<n-1; i++)
        fprintf(fp,"%d, ",rand()%nMax);
    fprintf(fp,"%d",rand()%nMax);
    fclose(fp);
}


