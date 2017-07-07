#ifndef _MATRIX
#define _MATRIX

#include <string>
#include <fstream>
#include <iostream>

using std::string;
using std::ifstream;
using std::ofstream;
using std::ostream;
using std::cout;
using std::endl;

class ColVector;

class Matrix
{

  public:
    unsigned m;
    unsigned n;

    Matrix(unsigned rows, unsigned cols);
    Matrix(unsigned rows, unsigned cols, int* vals);
    Matrix(string filename);

    int getRowCount() {return m;};
    int getColumnCount() {return n;};
    int* getValueBuffer() const {return values;};
    void print(); // make this compatible with ostream stuff..
    void readFromFile(string filename, bool vector=false);
    void setValueBuffer(int *newValues);
    void writeToFile(string filename);

    int operator()(unsigned row, unsigned col) const {return values[col + n*row];};
    int& operator()(unsigned row, unsigned col) {return values[col + n*row];};
    Matrix operator*(Matrix& m);
    ColVector operator*(ColVector& v);

    // holy trinity: dtor, copy ctor, assigment operator
    ~Matrix();
    Matrix(const Matrix& m);
    Matrix operator=(const Matrix& m) {return Matrix(m);};


  private:
    int *values;
};

// a COLUMN vector
class ColVector: public Matrix
{
  public:
    ColVector(int n);
    ColVector(int n, int* vals);
    ColVector(string filename);
    int getCount() {return getRowCount();};
    int operator()(unsigned row) const {return getValueBuffer()[row];};
    int& operator()(unsigned row) {return getValueBuffer()[row];};
};


#endif
