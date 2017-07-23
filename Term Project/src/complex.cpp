#include "complex.h"

template <class T>
Complex<T>::Complex(T re, T im)
{
  real = re;
  imag = im;
}

template <class T>
T Complex<T>::mag2() const
{
  return real*real+imag*imag;
}

template <class T>
Complex<T> Complex<T>::operator+(const T & c)
{
  return Complex<T>(real + c, imag);
}



template <class T>
Complex<T> Complex<T>::operator+(const Complex<T> & other)
{
  Complex<T> newC(real + other.real, imag + other.imag);
  return newC;
}

template <class T>
Complex<T>& Complex<T>::operator+=(const T & other)
{
  real += other;
  return *this;
}

template <class T>
Complex<T>& Complex<T>::operator+=(const Complex & other)
{
  real += other.real;
  imag += other.imag;
  return *this;
}



template <class T>
Complex<T> Complex<T>::operator-(const T & c)
{
  return Complex(real - c, imag);
}

template <class T>
Complex<T> Complex<T>::operator-(const Complex & other)
{
  Complex<T> newC(real - other.real, imag - other.imag);
  return newC;
}

template <class T>
Complex<T>& Complex<T>::operator-=(const T & other)
{
  real -= other;
  return *this;
}

template <class T>
Complex<T>& Complex<T>::operator-=(const Complex & other)
{
  real -= other.real;
  imag -= other.imag;
  return *this;
}


template <class T>
Complex<T> Complex<T>::operator*(const T & c)
{
    return Complex(real*c, imag*c);
}

template <class T>
Complex<T> Complex<T>::operator*(const Complex & other)
{
    T re = this->real*other.real - this->imag*other.imag;
    T im = this->real*other.imag + this->imag*other.real;
    return Complex<T>(re, im);
}

template <class T>
Complex<T>& Complex<T>::operator*=(const T & other)
{
  real *= other;
  imag *= other;
  return *this;
}

template <class T>
Complex<T>& Complex<T>::operator*=(const Complex & other)
{
  T re = this->real*other.real - this->imag*other.imag;
  T im = this->real*other.imag + this->imag*other.real;
  real = re;
  imag = im;
  return *this;
}


template <class T>
Complex<T> Complex<T>::operator/(const T & c)
{
    return Complex<T>(real/c, imag/c);
}

template <class T>
Complex<T> Complex<T>::operator/(const Complex & other)
{
    Complex<T> newC = (*this)*(!other);
    newC /= other.mag2();
    return newC;
}

template <class T>
Complex<T>& Complex<T>::operator/=(const T & c)
{
    real /= c;
    imag /= c;
    return *this;
}

template <class T>
Complex<T>& Complex<T>::operator/=(const Complex & other)
{
    Complex<T> newC = *this/other;
    real = newC.real;
    imag = newC.imag;
    return *this;
}


template <class T>
Complex<T>& Complex<T>::operator=(const Complex & other)
{
  real = other.real;
  imag = other.imag;
  return *this;
}

template <class T>
bool Complex<T>::operator==(const Complex & other)
{
  return (real==other.real) && (imag==other.imag);
}

template <class T>
Complex<T> Complex<T>::operator!() const
{
  return Complex<T>(real, -imag);
}


// force compilation of float and double methods
// other templated versions will cause undefined reference linker errors
// since implementation is separate from declaration (h file)
// That's ok though, since we really only want float and double versions anyhow...
template class Complex<float>;
template class Complex<double>;
