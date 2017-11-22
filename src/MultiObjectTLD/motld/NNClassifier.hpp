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

#ifndef NNCLASSIFIER_H
#define NNCLASSIFIER_H

//#include <iostream>
//#include <fstream>
#include <vector>
#include "Matrix.hpp"
#include "Histogram.hpp"

/// Data structure representing nearest neighbor patches with color histograms
class NNPatch
{
public:
  /// The patch (will be normalized to zero mean)
  Matrix patch;
  /// The average of the grayscale values (to enable reconstruction)
  float avg;
  /// Saves the squared norm of the grayscale values to speed up computation later
  float norm2;
  /// A normalized color histogram or NULL if not used
  float* histogram;
  /// Default (empty) constructor
  NNPatch() : patch(), avg(0), norm2(1), histogram(NULL){}
  /// Copy constructor
  NNPatch(const NNPatch& copyFrom);
  /// Constructor providing only a patch
  NNPatch(const Matrix& curPatch);  
  /// Constructor creating the color histogram
  NNPatch(const Matrix& curPatch, const ObjectBox& bbox, 
          const unsigned char * rgb = NULL, const int w = 0, const int h = 0);
  /// Constructor extracting the patch and the color histogram out of the image
  NNPatch(const ObjectBox& bbox, const Matrix& curImage, const int patchSize,
          const unsigned char * rgb = NULL, const int w = 0, const int h = 0);
  /// Constructor for loading from file
  NNPatch(std::ifstream & inputStream, const int patchSize);
  /// Destructor
  ~NNPatch();
  /// Copy operator
  NNPatch& operator=(const NNPatch& copyFrom);
  /// Method for saving to file
  void saveToStream(std::ofstream & outputStream) const;
};

/** @brief The nearest neighbor classifier is invoked at the top level to evaluate detections.
 */
class NNClassifier
{
public:
  /// Constructor
  NNClassifier(int width, int height, int patchSize, bool useColor = true, bool allowFastChange = false);
  /// Constructor for loading from file
  NNClassifier(std::ifstream & inputStream);
  /// Returns the confidence of a given patch with respect to a certain class.
  double getConf(const NNPatch& patch, int objId = 0, bool conservative = false) const;
  /// Returns the confidence of a given patch while subsequently computing and saving the color histogram if needed.
  double getConf(NNPatch& patch, int objId, bool conservative,
                  const ObjectBox& bbox, const unsigned char * rgb, int w, int h) const;
  /// Trains a new patch to the classifier if it is considered "new" enough.
  bool trainNN(const NNPatch& patch, int objId = 0, bool positive = true, bool tmp = false);
  /// Initializes a new object class with the given patch.
  void addObject(const NNPatch& patch);
  void deleteObject(int personID);
  /// Returns a pointer to positive patches (intended for drawing).
  const std::vector<std::vector<NNPatch> > * getPosPatches() const;
  /// Returns a pointer to negative patches (intended for drawing).
  const std::vector<NNPatch> * getNegPatches() const;
  /// Removes previously added warps (rotated patches) from positive list.
  void removeWarps();
  /// Saves the classifier (i.e. the patches) to file.
  void saveToStream(std::ofstream & outputStream) const;
  
private:
  int ivWidth;
  int ivHeight;
  int ivPatchSize;
  static Histogram * ivHistogram;
  std::vector<std::vector<NNPatch> > ivPosPatches;
  std::vector<NNPatch> ivNegPatches;
  std::vector<char> ivWarpIndices;
  bool ivUseColor, ivAllowFastChange;
  double getConf(const float* patch, float norm2 = 1.0f, int objId = 0, bool conservative = false) const;
  double crossCorr(const float* patchA, const float* patchB, float denom = 1) const;
  double cmpHistograms(const float* h1, const float* h2) const;
};

#endif //NNCLASSIFIER_H
