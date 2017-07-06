#include "matrix.h"


Matrix::Matrix(unsigned rows, unsigned cols)
{
  m = rows;
  n = cols;
  values = new int[rows*cols];
}


Matrix::Matrix(unsigned rows, unsigned cols, int* vals)
{
  m = rows;
  n = cols;
  values = vals;
}


Matrix::Matrix(const Matrix& mat)
{
  n = mat.n;
  m = mat.m;
  values = new int[m*n];
  for(unsigned irow=0; irow<m; irow++)
    for(unsigned jcol=0; jcol<n; jcol++)
      values[jcol + n*irow] = mat(irow, jcol);
}

// no error checking in here. Assumes file contains exactly what it says it does.
Matrix::Matrix(string filename)
{
  readFromFile(filename);
}

void Matrix::setValueBuffer(int* newValues)
{
  delete[] values;
  values = newValues;
}

void Matrix::readFromFile(string filename, bool vector)
{

  // open file
  ifstream infile(filename.c_str());

  // if vector, on pass through to get size
  if(vector)
  {
    m = 0;
    n = 1;
    int tmp;
    while( (infile >> tmp) )
      m++;
    infile.clear();
    infile.seekg(0);
  }
  else  // if NOT a vector, just read size of matrix
    infile >> m >> n;

  // read data
  values = new int[m*n];
  for(int i=0; i<m; i++)
    for(int j=0; j<n; j++)
      infile >> values[j + n*i];

  // close file
  infile.close();

}

Matrix& Matrix::operator*(Matrix& B)
{
  Matrix A = *(this);

  if(A.n!=B.m)
    throw std::invalid_argument("Matrix dimension mismatch.");

  Matrix &C = *(new Matrix(A.m,B.n));

  for(int irow=0; irow<A.m; irow++)
    for(int jcol=0; jcol<B.n; jcol++)
    {
      C(irow,jcol) = 0;
      for(int k=0; k<A.n; k++)
        C(irow,jcol) += A(irow,k)*B(k,jcol);
    }

    return C;
}


void Matrix::print()
{
  for(int i=0; i<m; i++)
  {
    cout << "[";
    for(int j=0; j<n; j++)
      cout << values[j+n*i] << ", ";
    cout << "\b\b]\n";
  }
}


Matrix::~Matrix()
{
  cout << "Matrix dtor!!!\n";
  delete[] values;
}


ColVector::ColVector(int n) : Matrix(n,1)
{}

ColVector::ColVector(int n, int* vals) : Matrix(n,1,vals)
{}

ColVector::ColVector(string filename) : Matrix(0,0)
{
  readFromFile(filename, true);
}
