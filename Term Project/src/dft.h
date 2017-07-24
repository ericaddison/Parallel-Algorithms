#include <complex>
#include <cmath>
#include <vector>

using std::complex;
using namespace std::literals::complex_literals;

typedef std::vector<complex<double>> cvec;

class dft
{

  public:
    dft(int n);
    ~dft();
    void forward(cvec& out, const cvec& in);
    void reverse(cvec& out, const cvec& in);

  private:
    int n;
    std::vector<cvec> w;
    double PI = acos(-1);
};
