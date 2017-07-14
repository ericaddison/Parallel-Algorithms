#include "hw3.h"
#include "ParallelHyperQuickSorter.h"

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
  if(world_rank==0)
    x.readFromFile(vectorFile);

  // create sorter object, sort, write to file
  ParallelHyperQuickSorter phqs(MPI_COMM_WORLD, x);
  phqs.sort();
  phqs.writeSortedArrayToFile("sortedArray.txt");

  if(world_rank==0)
    cout << "\nSorted array written to file ./sortedArray.txt\n\n";


  // cleanup
  MPI_Finalize();

}
