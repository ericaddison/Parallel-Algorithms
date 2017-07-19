#include "hw3.h"
#include "ParallelHyperQuickSorter.h"


/**
 * main
 * Main function for HW3, problem 2. Performs Parallel HyperQuickSort
 * of a list of numbers read in from text file specified as a command line
 * argument. Problem is implemented with a class to contain all sorting logic.
 */
int main(int argc, char** argv)
{

  // MPI init
  MPI_Init(NULL, NULL);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // check command line args
  if(argc!=2)
  {
    if(world_rank==0)
      cerr << "\nUsage: " << argv[0] << " <input file>\n\n";
    MPI_Finalize();
    return 0;
  }

  // get command line args
  string vectorFile(argv[1]);

  // rank0 read files and broadcast vector size
  ColVector x(0);
  ColVector trueResult(0);
  if(world_rank==0)
  {
    x.readFromFile(vectorFile);
    trueResult = x;
    quickSort(trueResult);
  }

  // create sorter object, sort, write to file
  ParallelHyperQuickSorter phqs(MPI_COMM_WORLD);
  phqs.sort(&x);

  // write to file
  if(world_rank==0)
  {
    x.writeToFile("sortedArray.txt");
    cout << "\nSorted array written to file ./sortedArray.txt\n";

    bool resultCorrect = x.equals(trueResult);
    if(resultCorrect)
      cout << "\nMPI result matches sequential result\n\n";
    else
      cout << "\nMPI result does NOT match sequential result\n\n";
  }


  // cleanup
  MPI_Finalize();

}
