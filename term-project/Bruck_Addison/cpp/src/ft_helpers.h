#ifndef _FT_HELPERS_H
#define _FT_HELPERS_H

#include <complex>
#include <cmath>
#include <vector>
#include <valarray>
#include <iostream>
#include <fstream>
#include <string>

using std::complex;
using std::string;
using std::ifstream;
typedef std::valarray< complex<double> > carray;
typedef complex<double> cdouble;

enum direction { FORWARD, REVERSE };
const double PI = acos(-1);

bool isPow2(int n);
void checkSize(int n);
int bitReverse(unsigned int b, int d);
void bitReverse(carray& x);
std::vector<double> readFile(string filename);

#endif
