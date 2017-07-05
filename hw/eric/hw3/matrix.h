
class Matrix
{

  public:
    Matrix(unsigned rows, unsigned cols);
    int getRowCount() {return m;};
    int getColumnCount() {return n;};
    int* getValueBuffer() const {return values;};
    int operator()(unsigned row, unsigned col) const {return values[col + n*row];};
    int& operator()(unsigned row, unsigned col) {return values[col + n*row];};

    // holy trinity: dtor, copy ctor, assigment operator
    ~Matrix();
    Matrix(const Matrix& m);
    Matrix& operator=(const Matrix& m) {return *(new Matrix(m));};


  private:
    unsigned m;
    unsigned n;
    int *values;
};

// a COLUMN vector
class Vector: Matrix
{
  public:
    Vector(int n);
    int getCount() {return this->getRowCount();};
    int operator()(unsigned row) const {return getValueBuffer()[row];};
    int& operator()(unsigned row) {return getValueBuffer()[row];};
};
