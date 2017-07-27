#ifndef _FT_HELPERS_H
#define _FT_HELPERS_H

#include <complex>
#include <cmath>
#include <valarray>
#include <iostream>

using std::complex;
typedef std::valarray< complex<double> > carray;
typedef complex<double> cdouble;

enum direction { FORWARD, REVERSE };
const double PI = acos(-1);

bool isPow2(int n);
void checkSize(int n);
int bitReverse(unsigned int b, int d);
void bitReverse(carray& x);

#endif
