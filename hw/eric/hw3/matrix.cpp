#include "matrix.h"


Matrix::Matrix(unsigned rows, unsigned cols)
{
  m = rows;
  n = cols;
  values = new int[rows*cols];
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

Matrix::~Matrix()
{
  delete[] values;
}


Vector::Vector(int n) : Matrix(n,1)
{}
