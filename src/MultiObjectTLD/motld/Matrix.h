/* Copyright (C) 2012 Christian Lutz, Thorsten Engesser
 * 
 * This file is part of motld
 * 
 * Some parts of this implementation are based
 * on materials to a lecture by Thomas Brox
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


#ifndef MATRIX_H
#define MATRIX_H

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <queue>
#include <stack>
#include <vector>
#include <algorithm>
#ifdef GNU_COMPILER
  #include <strstream>
#else
  #include <sstream>
#endif

#ifndef PI
#define PI 3.1415926536
#endif

#ifndef round
#define round(x) floor(x + 0.5)
#endif

#define MAXF(a,b,c,d) MAX(MAX(a,b),MAX(c,d))
#define MINF(a,b,c,d) MIN(MIN(a,b),MIN(c,d))
#ifndef MIN
#define MIN(a,b) (a < b ? a : b)
#define MAX(a,b) (a > b ? a : b)
#endif

/// datastructure linking objects to their (possible) location
struct ObjectBox
{
  /// x-component of top left coordinate
  float x;
  /// y-component of top left coordinate
  float y;
  /// width of the image section
  float width;
  /// height of the image section
  float height;
  /// identifies object, which is represented by ObjectBox
  int objectId;
};

/// datastructure for images (greyscale or single color)
class Matrix {
public:
  /// Default constructor
  inline Matrix();
  /// Constructor
  inline Matrix(const int width, const int height);
  /// Copy constructor
  Matrix(const Matrix& copyFrom);
  /// Constructor with implicit filling
  Matrix(const int width, const int height, const float value);
  /// Destructor
  virtual ~Matrix();

  /// fills the matrix from a char-array (size has to be already set)
  void copyFromCharArray(unsigned char * source);
  /// fills the matrix from a float-array (for a given size)
  void copyFromFloatArray(float * source, int srcwidth, int width, int height);
  /// fills the matrix from a sub-part of a float-array
  void copyFromFloatArray(const float * const source, int srcwidth, int srcheight, int x, int y, int width, int height);
  /// Creates a grayscale matrix out of r, g, and b matrices
  void fromRGB(const Matrix& rMatrix, const Matrix& gMatrix, const Matrix& bMatrix);
  /// Creates a grayscale matrix out of an array [r0, r1, ..., g0, g1, ..., b0, b1, ...]
  void fromRGB(unsigned char * source);
  /// Computes derivative in x direction (result will be in result)
  void derivativeX(Matrix& result) const;
  /// Computes derivative in y direction (result will be in result)
  void derivativeY(Matrix& result) const;
  /// Applies 3x3 Scharr filter in x direction (result will be in result)
  void scharrDerivativeX(Matrix& result) const;
  /// Applies 3x3 Scharr filter in y direction (result will be in result)
  void scharrDerivativeY(Matrix& result) const;
  /// Applies 3x3 Sobel filter in x direction (result will be in result)
  void sobelDerivativeX(Matrix& result) const;
  /// Applies 3x3 Sobel filter in y direction (result will be in result)
  void sobelDerivativeY(Matrix& result) const;
  /// Applies a Gaussian filter
  void gaussianSmooth(const float sigma, const int filterSize = 0);
  /// Saves the matrix as a picture in pgm-Format
  void writeToPGM(const char *filename) const;
  /// Returns a patch around the central point using bilinear interpolation
  Matrix getRectSubPix(float centerx, float centery, int width, int height) const;
 
  /// Changes the size of the matrix, data will be lost
  void setSize(int width, int height);
  /// Downsamples image to half of its size (result will be in result)
  void halfSizeImage(Matrix& result) const;
  /// Downsamples the matrix
  void downsample(int newWidth, int newHeight);
  /// Downsamples the matrix using bilinear interpolation
  void downsampleBilinear(int newWidth, int newHeight);  
  /// Upsamples the matrix
  void upsample(int newWidth, int newHeight);
  /// Upsamples the matrix using bilinear interpolation
  void upsampleBilinear(int newWidth, int newHeight);
  /// Scales the matrix (includes upsampling and downsampling)
  void rescale(int newWidth, int newHeight);
  
  /// Fills the matrix with the value value (see also operator =)
  void fill(const float value);
  /// Copies a rectangular part from the matrix into result, the size of result will be adjusted
  void cut(Matrix& result,const int x1, const int y1, const int x2, const int y2);
  /// Clips values that exceed the given range
  void clip(float aMin, float aMax);
  /// Inverts a 3x3 matrix
  void inv3();

  // Some drawing utilities
  /// Draws a line into the image
  void drawLine(int x1, int y1, int x2, int y2, float value = 255);
  /// Draws a Cross
  void drawCross(int x, int y, int value = 255, int crossSize = 1);
  /// Draws an ObjectBox into the image
  void drawBox(ObjectBox b, int value = 255);
  /// Draws a dashed ObjectBox into the image
  void drawDashedBox(ObjectBox b, int value = 255, int dashLength = 3, bool dotted = false);
  /// Draws a NN-Patch at position (x,y)
  void drawPatch(const Matrix & b, int x, int y, float avg = 0);
  /// Draws a histogram at position (x,y)
  void drawHistogram(const float * histogram, int x, int y, int value = 255, int nbins = 7, int psize = 15);
  /// Prints a number at position (x,y)
  void drawNumber(int x, int y, int n, int value = 255);
    
  /// Gives full access to matrix values
  inline float& operator()(const int ax, const int ay) const;
  /// Fills the matrix with the value value (equivalent to fill())
  inline Matrix& operator=(const float value);
  /// Copies the matrix copyFrom to this matrix (size of matrix might change)
  Matrix& operator=(const Matrix& copyFrom);
  /// Adds a constant to the matrix
  Matrix& operator+=(const float value);
  /// Multiplication with a scalar
  Matrix& operator*=(const float value);

  /// Returns the average value
  float avg() const;
  /// Returns the squared norm (i.e. sum of squared values)
  float norm2() const;
  /// Returns the width of the matrix
  inline int xSize() const;
  /// Returns the height of the matrix
  inline int ySize() const;
  /// Returns the size (width*height) of the matrix
  inline int size() const;
  /// Gives access to the internal data representation
  inline float* data() const;
  
  /// Performs an affine warping of an image section
  Matrix affineWarp(const Matrix & t, const ObjectBox & b, const bool & preservear) const;
  /// Creates a warp matrix for scaling / roatating
  static Matrix createWarpMatrix(const float& angle, const float& scale);
  /// Creates an Integral Image
  float* createSummedAreaTable() const;
  /// Creates an Integral Image and an Integral Image of squared values
  float** createSummedAreaTable2() const;
  
protected:
  int ivWidth, ivHeight;
  float *ivData;
};

/// Matrix product
Matrix operator*(const Matrix& m1, const Matrix& m2);
/// Provides basic output functionality (only appropriate for small matrices)
std::ostream& operator<<(std::ostream& aStream, const Matrix& aMatrix);

/// Outputs an RGB image in PPM format
void writePPM(const char* filename, const Matrix& rMatrix, const Matrix& gMatrix, const Matrix& bMatrix);

#endif
