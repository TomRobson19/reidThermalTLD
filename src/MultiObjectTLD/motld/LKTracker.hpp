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

#ifndef LKTracker_H
#define LKTracker_H
  
#include "Matrix.hpp"
#include <vector>
#include <algorithm>
#include <math.h>

#define KERNEL_WIDTH 2
#define MAX_PYRAMID_LEVEL 5
/// number of iterations in each pyramid level
#define LK_ITERATIONS 30
/// number of tracking points in the uniform grid
#define GRID_SIZE_X 10
#define GRID_SIZE_Y 10
/// relative padding to borders of bounding box
/// @note original TLD uses an absolute padding of 5px
#define GRID_PADDING 0.15
/// number of (additional) points chosen by cornerness measure (0 = disable => more efficient)
#define N_CORNERNESS_POINTS 0
/// absolute padding (used for cornerness points only)
#define GRID_MARGIN 3
#define KERNEL_SIZE ((KERNEL_WIDTH*2+1)*(KERNEL_WIDTH*2+1))
  
/** @brief This class contains the "short term tracking" part of the algorithm.
 */
class LKTracker
{
public:
  /// Constructor
  LKTracker(int width, int height) : ivWidth(width), ivHeight(height), 
      ivPrevPyramid(NULL), ivIndex(1) {};
  /// Destructor
  ~LKTracker() {delete ivPrevPyramid;};
  /// Sets up the internal image pyramid
  void initFirstFrame(unsigned char * img);
  /// Sets up the internal image pyramid
  void initFirstFrame(const Matrix& img);
  /** @brief Computes the optical flow for each object
   *  @param bbox List of current object boxes, they are replaced with the new boxes
   *  @param isDefined Must have the same size as @b bbox. True for each object that is 
   *    currently defined and should be tracked. Is set to false if tracking failed.
   */
  void processFrame(const Matrix& curImage, std::vector<ObjectBox>& bbox, std::vector<bool>& isDefined); 
  /// An adapter for the single object case
  bool processFrame(const Matrix& curImage, ObjectBox& bbox, bool dotracking = true); 
  /// A list of points [x0,y0,...,xn,yn] that where considered as inliers in the last iteration
  const std::vector<int> * getDebugPoints() const { return &ivDebugPoints; };
  
private:
  /// Internal representation for an image pyramid
  struct LKPyramid
  {
    std::vector<Matrix> I,Ix,Iy;
    LKPyramid(){};
    LKPyramid(int nLevels){
      I = std::vector<Matrix>(nLevels);
      Ix = std::vector<Matrix>(nLevels);
      Iy = std::vector<Matrix>(nLevels); 
    };
  };
  /// Simple representation for 2D (sub pixel) image points
  struct Point2D
  {
      float x, y;
  };
  int ivWidth;
  int ivHeight;
  LKPyramid* ivPrevPyramid;
  int ivIndex;
  std::vector<int> ivDebugPoints;
  /** Computes median of a vector
   * @note changes order of vector-elements! */
  inline float median(std::vector<float> * vec, bool compSqrt = false) const;
  /** Computes normalized cross correlation
   * @details defined as: @f[NCC(A,B):=\frac{\sum_{x,y}(A(x,y)-\bar{A})(B(x,y)-\bar{B})}
    *    {\sqrt{\sum_{x,y}(A(x,y)-\bar{A})^2\sum_{x,y}(B(x,y)-\bar{B})^2}} @f]
   */
  inline double NCC(const Matrix& aMatrix, const Matrix& bMatrix) const;
  /** Computes optical flow for each tracking point. 
   * @details Based on the technical report "Pyramidal Implementation of the 
   *  Lucas Kanade Feature Tracker: Description of the algorithm" by Jean-Yves Bouguet */
  inline void pyramidLK(const LKPyramid *prevPyramid, const LKPyramid *curPyramid, 
                        const Point2D *prevPts, Point2D *nextPts, 
                        char *status, int count) const;
};


#endif 
