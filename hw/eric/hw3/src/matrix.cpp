#include "matrix.h"



/**
 * Matrix
 * Matrix constructor to specify number of rows and columns.
 *
 * @param rows number of rows in matrix
 * @param cols number of columns in matrix
 */
Matrix::Matrix(unsigned rows, unsigned cols)
{
  m = rows;
  n = cols;
  values = new int[rows*cols];
}



/**
 * Matrix
 * Matrix constructor to specify number of rows and columns and data.
 * It is assumed that the provided data pointer has been properly allocated.
 *
 * @param rows number of rows in matrix
 * @param cols number of columns in matrix
 * @param vals pointer to array of values
 */
Matrix::Matrix(unsigned rows, unsigned cols, int* vals)
{
  m = rows;
  n = cols;
  values = vals;
}



/**
 * Matrix
 * Copy constructor.
 *
 * @param mat matrix to copy
 */
Matrix::Matrix(const Matrix& mat)
{
  n = mat.n;
  m = mat.m;
  values = new int[m*n];
  std::copy(mat.values, mat.values + m*n, values);
}



/**
 * operator=
 * Assignment overload
 *
 * @param mat incoming matrix
 * @return reference to this matrix
 */
Matrix& Matrix::operator=(const Matrix& mat)
{
  n = mat.n;
  m = mat.m;
  delete[] values;
  values = new int[m*n];
  std::copy(mat.values, mat.values + m*n, values);
  return *this;
}



/**
 * Matrix
 * Contructor for reading a matrix from a text file.
 *
 * @param filename path to text file
 */
Matrix::Matrix(string filename)
{
  readFromFile(filename);
}



/**
 * setValueBuffer
 * Set a new buffer to hold data values.
 * Old buffer is free'd.
 * Assumed that new buffer has been properly allocated.
 *
 * @param newValues pointer to new data buffer
 * @param newM new number of rows
 *_@param newN new number of columns
 */
void Matrix::setValueBuffer(int* newValues, int newM, int newN)
{
  delete[] values;
  values = newValues;
  m = newM;
  n = newN;
}



/**
 * readFromFile
 * Read values from a text file into the matrix.
 *
 * @param filename path to text file
 * @param vector whether the calling object is a vector. Default=false
 */
void Matrix::readFromFile(string filename, bool vector)
{

  // open file
  ifstream infile(filename.c_str());

  // if vector, one pass through to get size
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



/**
 * fill
 * Fill the matrix with a given value.
 *
 * @param value the value to fill the matix with
 */
void Matrix::fill(int value)
{
  for(int i=0; i<m; i++)
    for(int j=0; j<n; j++)
      values[j+i*n] = value;
}



/**
 * operator*
 * Operator overload for matrix multiplication
 *
 * @param B right-multiplied matrix
 * @return matrix product
 */
Matrix Matrix::operator*(Matrix& B)
{
  Matrix A = *(this);

  if(A.n!=B.m)
    throw std::invalid_argument("Matrix dimension mismatch.");

  Matrix C(A.m,B.n);

  for(int irow=0; irow<A.m; irow++)
    for(int jcol=0; jcol<B.n; jcol++)
    {
      C(irow,jcol) = 0;
      for(int k=0; k<A.n; k++)
        C(irow,jcol) += A(irow,k)*B(k,jcol);
    }

    return C;
}



/**
 * operator*
 * Operator overload for matrix-vector multiplication. Returns a ColVector
 *
 * @param B right-multiplied vector
 * @return vector product
 */
ColVector Matrix::operator*(ColVector& B)
{
  Matrix A = *(this);

  if(A.n!=B.m)
    throw std::invalid_argument("Matrix dimension mismatch.");

  ColVector C(A.m);

  for(int irow=0; irow<A.m; irow++)
    {
      C(irow) = 0;
      for(int k=0; k<A.n; k++)
        C(irow) += A(irow,k)*B(k);
    }

  return C;
}



/**
 * equals
 * Check if this matrix is equal in dimension and values to another matrix
 *
 * @param y the other matrix
 * @return boolean whether dimensions and values are equal
 */
bool Matrix::equals(Matrix& y)
{
  if(m!=y.m || n!=y.n)
    return false;

  for(int i=0; i<m*n; i++)
    if(values[i]!=y.values[i])
      return false;

  return true;
}



/**
 * print
 * Convenience function for printing matrix values. Values are printed
 * in a 2D array.
 *
 * @param os ostream reference to print to
 */
void Matrix::print(ostream& os)
{
  for(int i=0; i<m; i++)
  {
    os << "[";
    for(int j=0; j<n-1; j++)
      os << values[j+n*i] << ", ";
    os << values[n-1+n*i] << "]\n";
  }
}



/**
 * printLinear
 * Convenience function for printing matrix values. Values are printed
 * sequentially.
 *
 * @param os ostream reference to print to
 */
void Matrix::printLinear(ostream& os)
{
  for(int i=0; i<n*m; i++)
    os << values[i] << " ";
}



/**
 * writeToFile
 * Write matrix values to a text file. Vectors are always written
 * out as a space separated list.
 *
 * @param filename path to output file
 */
void Matrix::writeToFile(string filename)
{
  ofstream outFile;
  outFile.open(filename.c_str());
  bool isVector = (m==1 || n==1);
  if(!isVector)  // if not a vector
    outFile << m << " " << n;

  for(int irow=0; irow<m; irow++)
  {
      outFile << ( isVector ? "" : "\n" );
    for(int jcol=0; jcol<n; jcol++)
      outFile << values[jcol + n*irow] << " ";
  }

  outFile.close();
}



/**
 * ~Matrix
 * Matrix destructor. Frees value buffer.
 *
 */
Matrix::~Matrix()
{
  delete[] values;
}



/**
 * ColVector
 * The following methods are simple overrides of Matrix methods for
 * Convenience.
 */

ColVector::ColVector(int n) : Matrix(n,1)
{}

ColVector::ColVector(int n, int* vals) : Matrix(n,1,vals)
{}

ColVector::ColVector(string filename) : Matrix(0,0)
{
  readFromFile(filename);
}

void ColVector::readFromFile(string filename)
{
  Matrix::readFromFile(filename,true);
}
