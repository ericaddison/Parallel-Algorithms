#include <gtest/gtest.h>
#include <complex.h>
#include<iostream>

float floatEps = 1e-6;
double doubleEps = 1e-12;

class ComplexTest : public testing::Test
{
};


TEST(ComplexTest_Instantiate, zero)
{
  Complex<> c;
  EXPECT_NEAR(c.real, 0, floatEps);
  EXPECT_NEAR(c.imag, 0, floatEps);
}

TEST(ComplexTest_Instantiate, realValue)
{
  Complex<> c(1.234,0);
  EXPECT_NEAR(c.real, 1.234, floatEps);
  EXPECT_NEAR(c.imag, 0, floatEps);
}

TEST(ComplexTest_Instantiate, imagValue)
{
  Complex<> c(0, 1);
  EXPECT_NEAR(c.real, 0, floatEps);
  EXPECT_NEAR(c.imag, 1, floatEps);
}

TEST(ComplexTest_Instantiate, twoValues)
{
  Complex<> c(2, 1);
  EXPECT_NEAR(c.real, 2, floatEps);
  EXPECT_NEAR(c.imag, 1, floatEps);
}

TEST(ComplexTest_Instantiate, doubleComplex)
{
  Complex<double> c(2.123, -0.2345);
  EXPECT_NEAR(c.real, 2.123, doubleEps);
  EXPECT_NEAR(c.imag, -0.2345, doubleEps);
}

TEST(ComplexTest_Mag2, test1)
{
  Complex<float> c(1, -1);
  EXPECT_NEAR(c.mag2(), 2, floatEps);
}
