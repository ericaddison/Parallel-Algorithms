#include <gtest/gtest.h>
#include <complex.h>
#include<iostream>

float floatEps = 1e-6;
double doubleEps = 1e-12;

class ComplexTest : public testing::Test
{
};


//***************************************************
// Test Instantiation

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



//***************************************************
// Test Mag2

TEST(ComplexTest_Mag2, test1)
{
  Complex<float> c;
  EXPECT_NEAR(c.mag2(), 0, floatEps);
}

TEST(ComplexTest_Mag2, test2)
{
  Complex<float> c(2, -2);
  EXPECT_NEAR(c.mag2(), 8, floatEps);
}

TEST(ComplexTest_Mag2, test3)
{
  Complex<float> c(1, 0);
  EXPECT_NEAR(c.mag2(), 1, floatEps);
}


TEST(ComplexTest_Mag2, test1D)
{
  Complex<double> c;
  EXPECT_NEAR(c.mag2(), 0, doubleEps);
}

TEST(ComplexTest_Mag2, test2D)
{
  Complex<double> c(2, -2);
  EXPECT_NEAR(c.mag2(), 8, doubleEps);
}

TEST(ComplexTest_Mag2, test3D)
{
  Complex<double> c(1, 0);
  EXPECT_NEAR(c.mag2(), 1, doubleEps);
}


//***************************************************
// Test Addition

TEST(ComplexTest_Addition, addScalar)
{
  Complex<> c(0, 0);
  Complex<> d = c+1;
  EXPECT_NEAR(d.real, 1, floatEps);
  EXPECT_NEAR(d.imag, 0, floatEps);
}

TEST(ComplexTest_Addition, addScalarD)
{
  Complex<double> c(3, 10);
  Complex<double> d = c+10;
  EXPECT_NEAR(d.real, 13, doubleEps);
  EXPECT_NEAR(d.imag, 10, doubleEps);
}

TEST(ComplexTest_Addition, addComplex)
{
  Complex<> c(1, 2);
  Complex<> d(0.5, -0.5);
  Complex<> e = c+d;
  EXPECT_NEAR(e.real, 1.5, floatEps);
  EXPECT_NEAR(e.imag, 1.5, floatEps);
}

TEST(ComplexTest_Addition, addComplexD)
{
  Complex<double> c(1, 2);
  Complex<double> d(0.5, -0.5);
  Complex<double> e = c+d;
  EXPECT_NEAR(e.real, 1.5, doubleEps);
  EXPECT_NEAR(e.imag, 1.5, doubleEps);
}

TEST(ComplexTest_Addition, addToselfScalar)
{
  Complex<> c(1, 2);
  c += 10;
  EXPECT_NEAR(c.real, 11, floatEps);
  EXPECT_NEAR(c.imag, 2, floatEps);
}

TEST(ComplexTest_Addition, addToselfScalarD)
{
  Complex<double> c(1, 2);
  c += 10;
  EXPECT_NEAR(c.real, 11, doubleEps);
  EXPECT_NEAR(c.imag, 2, doubleEps);
}

TEST(ComplexTest_Addition, addToselfComplex)
{
  Complex<> c(1, 2);
  Complex<> d(1, 1);
  c += d;
  EXPECT_NEAR(c.real, 2, floatEps);
  EXPECT_NEAR(c.imag, 3, floatEps);
}

TEST(ComplexTest_Addition, addToselfComplexD)
{
  Complex<double> c(1, 2);
  Complex<double> d(1, 1);
  c += d;
  EXPECT_NEAR(c.real, 2, doubleEps);
  EXPECT_NEAR(c.imag, 3, doubleEps);
}



//***************************************************
// Test Subtraction

TEST(ComplexTest_Subtraction, subScalar)
{
  Complex<> c(0, 0);
  Complex<> d = c-1;
  EXPECT_NEAR(d.real, -1, floatEps);
  EXPECT_NEAR(d.imag, 0, floatEps);
}

TEST(ComplexTest_Subtraction, subScalarD)
{
  Complex<double> c(3, 10);
  Complex<double> d = c-10;
  EXPECT_NEAR(d.real, -7, doubleEps);
  EXPECT_NEAR(d.imag, 10, doubleEps);
}

TEST(ComplexTest_Subtraction, subComplex)
{
  Complex<> c(1, 2);
  Complex<> d(0.5, -0.5);
  Complex<> e = c-d;
  EXPECT_NEAR(e.real, 0.5, floatEps);
  EXPECT_NEAR(e.imag, 2.5, floatEps);
}

TEST(ComplexTest_Subtraction, subComplexD)
{
  Complex<double> c(1, 2);
  Complex<double> d(0.5, -0.5);
  Complex<double> e = c-d;
  EXPECT_NEAR(e.real, 0.5, doubleEps);
  EXPECT_NEAR(e.imag, 2.5, doubleEps);
}

TEST(ComplexTest_Subtraction, subToselfScalar)
{
  Complex<> c(1, 2);
  c -= 10;
  EXPECT_NEAR(c.real, -9, floatEps);
  EXPECT_NEAR(c.imag, 2, floatEps);
}

TEST(ComplexTest_Subtraction, subToselfScalarD)
{
  Complex<double> c(1, 2);
  c -= 10;
  EXPECT_NEAR(c.real, -9, doubleEps);
  EXPECT_NEAR(c.imag, 2, doubleEps);
}

TEST(ComplexTest_Subtraction, subToselfComplex)
{
  Complex<> c(1, 2);
  Complex<> d(1, 1);
  c -= d;
  EXPECT_NEAR(c.real, 0, floatEps);
  EXPECT_NEAR(c.imag, 1, floatEps);
}

TEST(ComplexTest_Subtraction, subToselfComplexD)
{
  Complex<double> c(1, 2);
  Complex<double> d(1, 1);
  c -= d;
  EXPECT_NEAR(c.real, 0, doubleEps);
  EXPECT_NEAR(c.imag, 1, doubleEps);
}


//***************************************************
// Test Multiplication

TEST(ComplexTest_Multiplication, multScalar)
{
  Complex<> c(1, 2);
  Complex<> d = c*2;
  EXPECT_NEAR(d.real, 2, floatEps);
  EXPECT_NEAR(d.imag, 4, floatEps);
}

TEST(ComplexTest_Multiplication, multScalarD)
{
  Complex<double> c(3, 10);
  Complex<double> d = c*10;
  EXPECT_NEAR(d.real, 30, doubleEps);
  EXPECT_NEAR(d.imag, 100, doubleEps);
}

TEST(ComplexTest_Multiplication, multComplex)
{
  Complex<> c(1, 2);
  Complex<> d(0.5, -0.5);
  Complex<> e = c*d;
  EXPECT_NEAR(e.real, 1.5, floatEps);
  EXPECT_NEAR(e.imag, 0.5, floatEps);
}

TEST(ComplexTest_Multiplication, multComplexD)
{
  Complex<double> c(1, 2);
  Complex<double> d(0.5, -0.5);
  Complex<double> e = c*d;
  EXPECT_NEAR(e.real, 1.5, doubleEps);
  EXPECT_NEAR(e.imag, 0.5, doubleEps);
}

TEST(ComplexTest_Multiplication, multToselfScalar)
{
  Complex<> c(1, 2);
  c *= 10;
  EXPECT_NEAR(c.real, 10, floatEps);
  EXPECT_NEAR(c.imag, 20, floatEps);
}

TEST(ComplexTest_Multiplication, multToselfScalarD)
{
  Complex<double> c(1, 2);
  c *= 10;
  EXPECT_NEAR(c.real, 10, doubleEps);
  EXPECT_NEAR(c.imag, 20, doubleEps);
}

TEST(ComplexTest_Multiplication, multToselfComplex)
{
  Complex<> c(1, 2);
  Complex<> d(1, 1);
  c *= d;
  EXPECT_NEAR(c.real, -1, floatEps);
  EXPECT_NEAR(c.imag, 3, floatEps);
}

TEST(ComplexTest_Multiplication, multToselfComplexD)
{
  Complex<double> c(1, 2);
  Complex<double> d(1, 1);
  c *= d;
  EXPECT_NEAR(c.real, -1, doubleEps);
  EXPECT_NEAR(c.imag, 3, doubleEps);
}


//***************************************************
// Test Division

TEST(ComplexTest_Division, divScalar)
{
  Complex<> c(2, 3);
  Complex<> d = c/2;
  EXPECT_NEAR(d.real, 1, floatEps);
  EXPECT_NEAR(d.imag, 1.5, floatEps);
}

TEST(ComplexTest_Division, divScalarD)
{
  Complex<double> c(3, 10);
  Complex<double> d = c/10;
  EXPECT_NEAR(d.real, 0.3, doubleEps);
  EXPECT_NEAR(d.imag, 1, doubleEps);
}

TEST(ComplexTest_Division, divComplex)
{
  Complex<> c(1, 2);
  Complex<> d(3, 4);
  Complex<> e = c/d;
  EXPECT_NEAR(e.real, 0.44, floatEps);
  EXPECT_NEAR(e.imag, 0.08, floatEps);
}

TEST(ComplexTest_Division, divComplexD)
{
  Complex<double> c(1, 2);
  Complex<double> d(3, -4);
  Complex<double> e = c/d;
  EXPECT_NEAR(e.real, -0.2, doubleEps);
  EXPECT_NEAR(e.imag, 0.4, doubleEps);
}

TEST(ComplexTest_Division, divToselfScalar)
{
  Complex<> c(1, 2);
  c /= 10;
  EXPECT_NEAR(c.real, 0.1, floatEps);
  EXPECT_NEAR(c.imag, 0.2, floatEps);
}

TEST(ComplexTest_Division, divToselfScalarD)
{
  Complex<double> c(1, 2);
  c /= 10;
  EXPECT_NEAR(c.real, 0.1, doubleEps);
  EXPECT_NEAR(c.imag, 0.2, doubleEps);
}

TEST(ComplexTest_Division, divToselfComplex)
{
  Complex<> c(1, 2);
  Complex<> d(1, 1);
  c /= d;
  EXPECT_NEAR(c.real, 1.5, floatEps);
  EXPECT_NEAR(c.imag, 0.5, floatEps);
}

TEST(ComplexTest_Division, divToselfComplexD)
{
  Complex<double> c(1, 2);
  Complex<double> d(1, -1);
  c /= d;
  EXPECT_NEAR(c.real, -0.5, doubleEps);
  EXPECT_NEAR(c.imag, 1.5, doubleEps);
}



//***************************************************
// Test Others

TEST(ComplexTest_Other, assignment)
{
  Complex<> c(1, 2);
  Complex<> d = c;
  EXPECT_NEAR(c.real, d.real, floatEps);
  EXPECT_NEAR(c.imag, d.imag, floatEps);
}

TEST(ComplexTest_Other, assignmentD)
{
  Complex<double> c(1, 2);
  Complex<double> d = c;
  EXPECT_NEAR(c.real, d.real, doubleEps);
  EXPECT_NEAR(c.imag, d.imag, doubleEps);
}

TEST(ComplexTest_Other, equality)
{
  Complex<> c(1, 2);
  Complex<> d(1, 2);
  EXPECT_TRUE(c==d);
}

TEST(ComplexTest_Other, equalityFail)
{
  Complex<double> c(1, 2);
  Complex<double> d(2, 3);
  EXPECT_FALSE(c==d);
}

TEST(ComplexTest_Other, conjugate)
{
  Complex<> c(1, 2);
  EXPECT_NEAR((!c).real,1,floatEps);
  EXPECT_NEAR((!c).imag,-2,floatEps);
}
