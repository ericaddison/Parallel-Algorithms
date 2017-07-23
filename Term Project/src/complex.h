
template <class T=float>
class Complex
{
  public:
    T real;
    T imag;

    Complex<T>(T re=0, T im=0)
    {real=re; imag=im;};

    T mag2();

    Complex<T> operator+(const T & other);
    Complex<T> operator+(const Complex<T> & other);
    Complex<T>& operator+=(const T & other);
    Complex<T>& operator+=(const Complex<T> & other);

    Complex<T> operator-(const T & other);
    Complex<T> operator-(const Complex<T> & other);
    Complex<T>& operator-=(const T & other);
    Complex<T>& operator-=(const Complex<T> & other);

    Complex<T> operator*(const T & other);
    Complex<T> operator*(const Complex<T> & other);
    Complex<T>& operator*=(const T & other);
    Complex<T>& operator*=(const Complex<T> & other);

    Complex<T> operator/(const T & other);
    Complex<T> operator/(const Complex<T> & other);
    Complex<T>& operator/=(const T & other);
    Complex<T>& operator/=(const Complex<T> & other);

    Complex<T>& operator=(const Complex<T> & other);
    bool operator==(const Complex<T> & other);
    Complex<T> operator!();
};
