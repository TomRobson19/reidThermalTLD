/* Copyright (C) 2012 Christian Lutz, Thorsten Engesser
 * 
 * This file is part of motld
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

#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <cmath>
#include <cstring>
#include <vector>
#include "Matrix.h"

#define MAX3(a,b,c) a > b ? (a > c ? a : c) : (b > c ? b : c);
#define MIN3(a,b,c) a < b ? (a < c ? a : c) : (b < c ? b : c);

#define NUM_BINS 7
#define GRAY_THRESHOLD 0.125

/// ultra discrete color histograms intended to be used as (weak) classifier asset
class Histogram {
public:
  /// get instance of histogram generating singleton
  static Histogram * getInstance();
  /// creates histogram from whole image
  float * getColorDistribution(const unsigned char * const rgb, const int & size) const;
  /// creates histogram from whole image
  float * getColorDistribution(const unsigned char * const rgb, const int & width, const int & height) const;
  /// creates histogram from image section
  float * getColorDistribution(const unsigned char * const rgb, const int & width, const int & height, const ObjectBox & box) const;
  /// creates a debug image with colors which are maped to same histogram value
  unsigned char * debugImage(const int & bin, int & sideLength) const;
  /// compares two histograms by performing a normalized chi-squared test on their average
  static float compareColorDistribution(const float * const hist1, const float * const hist2);
  
private:
  Histogram();
  ~Histogram();
  
  static void toHS(const float & r, const float & g, const float & b, float & h, float & s);
  static float chiSquareSym(const float * const distr1, const float * const distr2, const int & n);
  static float chiSquare(const float * const correctHistogram, const float * const toCheck, const int & n);
  
  static Histogram * ivInstance;
  unsigned char * ivLookupRGB;
};

#endif // HISTOGRAM_H
