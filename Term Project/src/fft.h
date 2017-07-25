#include <complex>
#include <cmath>
#include <vector>

using std::complex;
using namespace std::literals::complex_literals;
typedef std::vector<complex<double>> cvec;
typedef complex<double> cdouble;

enum direction { FORWARD, REVERSE };

class fft
{

  public:
    fft(int n);
    ~fft();
    void forward(cdouble* out, const cdouble* in);
    void reverse(cdouble* out, const cdouble* in);

  private:
    int n;
    std::vector<cvec> w;
    double PI = acos(-1);
    double v;
    void transform(cdouble* out, const cdouble* in, int N, int stride, direction dir);
};
