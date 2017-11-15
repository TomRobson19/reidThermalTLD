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


#ifndef FERNFILTER_H
#define FERNFILTER_H

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <map>

#ifdef WIN32
#include <time.h>
#else
#include <sys/time.h>
#endif

#include "Matrix.hpp"
#include "Utils.hpp"

#define USEMAP 1       // Default: 1 - 0 = use lookup table instead: experimental
#define USETBBP 1      // Default: 1 - 0 = use simple pixel comparison: experimental
#define USEFASTSCAN 0  // Default: 0 - 1 = scan only every 2nd box: experimental

#define TIMING 0

// some settings - don't change these!
#define CONFIDENCETHRESHOLD             0.7
#define POSOVERLAPTHRESHOLD             0.85
#define NEGOVERLAPTHRESHOLD             0.65
#define INITNEGOVERLAPTHRESHOLD         0.20
#define NUMNEGTRAININGEXAMPLES         100
#define VARIANCETHRESHOLDFACTOR        0.8
#define VARIANCETHRESHOLDDESCENDRATE   0.2
#define VARIANCEMINTHRESHOLD           100


/// defines settings for affine warps, which are used in the FernFilter update process
struct WarpSettings
{
  /// (currently unused)
  int num_closest;
  /// how many warps should be done
  int num_warps;
  /// (currently unused)
  float noise;
  /// max rotation angle in degrees
  float angle;
  /// max translation shift in per cent
  float shift;
  /// max relative scale in per cent
  float scale;
};

/// describes a detection by the FernFilter
struct FernDetection
{
  /// the object box (location, dimension, object id)
  ObjectBox box;
  /// patch as downscaled, squared image
  Matrix patch;
  /// confidence value associated with patch
  float confidence;
  /// fern features associated with patch
  int * featureData;
  // temporary values
  const void * ss;     // pointer to scan parameters used for this detection
  float * imageOffset; // pointer to image / sat position required to compute featureData
  /// ordering over FernDetections (using their confidence values)
  static bool fdBetter(FernDetection fd1, FernDetection fd2) { return fd1.confidence > fd2.confidence; }
};

/// used for learning and (re-)finding objects
class FernFilter
{
public:
  /// public constructor
  FernFilter(const int & width, const int & height, const int & numFerns,
             const int & featuresPerFern, const int & patchSize = 15, 
             const int & scaleMin = -10, const int & scaleMax = 11, const int & bbMin = 24);
  /// copy constructor
  FernFilter(const FernFilter & other);
  /// destructor
  ~FernFilter();
  /// introduces new objects from a list of object boxes and returns negative training examples
  const std::vector<Matrix> addObjects(const Matrix & image, const std::vector<ObjectBox>& boxes);
  /// scans fern structure for possible object matches using a sliding window approach
  const std::vector<FernDetection> scanPatch(const Matrix & image) const;
  /// updates the fern structure with information about the correct boxes
  const std::vector< Matrix > learn(const Matrix& image, const std::vector< ObjectBox >& boxes, bool onlyVariance = false);
  /// creates a FernFilter from binary stream (load procedure)
  static FernFilter loadFromStream(std::ifstream & inputStream);
  /// writes FernFilter into binary stream (save procedure)
  void saveToStream(std::ofstream & outputStream) const;
  /// changes input image dimensions (has to be applied with applyPreferences())
  void changeInputFormat(const int & width, const int & height);
  /// changes default size scan box dimensions (has to be applied with applyPreferences())
  void changeScanBoxFormat(const int & width, const int & height);
  /// changes sliding window preferences (has to be applied with applyPreferences())
  void changeScanSettings(const int & scaleMin, const int & scaleMax, const int & bb_min);
  /// applies changes made by changeInputFormat() changeScanBoxFormat(), changeScanSettings()
  void applyPreferences();
  /// changes settings for warping
  void changeWarpSettings(const WarpSettings & initSettings, const WarpSettings & updateSettings);
  
private:
  // Methods for feature extraction / fern manipulation etc.
  void createScaledMatrix(const Matrix& image, Matrix & scaled, float*& sat, float*& sat2, int scale) const;
  void createScaledMatrices(const Matrix& image, Matrix*& scaled, float**& sats, float**& sat2s) const;
  void varianceFilter(float * image, float * sat, float * sat2, int scale, std::vector<FernDetection> & acc) const;
  std::vector< Matrix > retrieveHighVarianceSamples(const Matrix& image, const std::vector< ObjectBox >& boxes);
  int* extractFeatures(const float * const imageOrSAT, int ** offsets) const;
  void extractFeatures(FernDetection & det) const;
  float calcMaxConfidence(int * features) const;
  float * calcConfidences(int * features) const;
  void addPatch(const int & objId, const int * const featureData, const bool & pos);
  void addPatch(const Matrix& scaledImage, const int& objId, const bool& pos);
  void addPatchWithWarps(const Matrix & image, const ObjectBox & box, const WarpSettings & ws, 
                         std::vector<Matrix> & op, const bool & pos, const bool & notOnlyVar = true);
  void addWarpedPatches(const Matrix & image, const ObjectBox & box, const WarpSettings & ws,
                        std::vector<Matrix> & op, const bool & pos);
  void clearLastDetections() const;
  
  // Methods for initialization
  int *** createFeatures();
  void initializeFerns();
  void computeOffsets();
  int ** computeOffsets(int width);
  void addObjectToFerns();
  
  // Helper
  int calcTableSize() const;
  void debugOutput() const;
  FernDetection copyFernDetection(const FernDetection & fd) const;
  
  struct Posteriors
  {
    int n;
    int p;
    float posterior;
  };

  struct Confidences
  { 
    float maxConf;
    std::map<int, Posteriors> posteriors;
  };

  struct ScanSettings
  {
    int width;
    int height;
    float boxw;
    float boxh;
    float pixw;
    float pixh;
    int * varianceIndizes;
    int ** offsets;
  };
  
  #if DEBUG && TIMING
  struct ScanTime
  {
    int time0ScaledImages;
    int time1Variance;
    int time2GenFeatures;
    int time3CoarseFilter;
    int time4FineFilter;
    int time5GenPatches;
  };
#endif
  
  // changeable input image dimensions
  int ivWidth;
  int ivHeight;
  
  // hard classifier configuration
  const int ivNumFerns;
  const int ivFeaturesPerFern;
  const int ivPatchSize;
  
  // helper classifier constants
  const int ivPatchSizeMinusOne;
  const int ivPatchSizeSquared;
  
  // changable scan settings
  int ivOriginalWidth;
  int ivOriginalHeight;
  int ivScaleMin;
  int ivScaleMax;
  int ivBBmin;
  WarpSettings ivInitWarpSettings;
  WarpSettings ivUpdateWarpSettings;
  
  // Fern Data
  int *** ivFeatures;
#if USEMAP
  std::map<int, Confidences> * ivFernForest;
#else
  std::vector<int**> ivNtable;
  std::vector<int**> ivPtable;
  std::vector<float**> ivTable;
  float **ivMaxTable;
#endif
  
  // further instance variables
  int ivNumObjects;
  int ivScanNoZoom;
  float ivVarianceThreshold;
  int ** ivPatchSizeOffsets;
  std::vector<ScanSettings> ivScans;
  std::vector<float> ivMinVariances;
  mutable std::vector<FernDetection> ivLastDetections;
  
  // some default structures
  static const WarpSettings cDefaultInitWarpSettings;
  static const WarpSettings cDefaultUpdateWarpSettings;
};

#endif //FERNCLASSIFIER_H
