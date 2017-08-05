#ifndef _FFT_H
#define _FFT_H

#include "ft_helpers.h"
#include <iostream>

// fft functions
void fft(carray& x);
void cilk_fft(carray& x);
void iter_fft(carray& x);
void cilk_iter_fft(carray& x);
void ifft(carray& x);


#endif
