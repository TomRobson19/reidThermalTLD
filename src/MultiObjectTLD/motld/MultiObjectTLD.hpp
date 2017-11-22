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

#ifndef MULTIOBJECTTLD_H
#define MULTIOBJECTTLD_H

//#define DEBUG 1
#define TIMING 0

#include <iostream>
#include <fstream>
#include <vector>
#include "LKTracker.hpp"
#include "FernFilter.hpp"
#include "NNClassifier.hpp"
#include "Matrix.hpp"

#define COLOR_MODE_GRAY 0
#define COLOR_MODE_RGB 1

/// defines concerning getStatus()
#define STATUS_LOST 0
#define STATUS_UNSURE 1
#define STATUS_OK 2

/// defines concerning writeDebugImage()
#define DEBUG_DRAW_DETECTIONS 1
#define DEBUG_DRAW_PATCHES 2
#define DEBUG_DRAW_CROSSES 4


#define ENABLE_CLUSTERING 1

/// Settings-structure that may be passed to the constructor of MultiObjectTLD
struct MOTLDSettings
{
    ///@brief mode in which the images are passed to MultiObjectTLD::processFrame().
    /// Supported modes are COLOR_MODE_GRAY (default) and COLOR_MODE_RGB
    int colorMode;
    /// width and height of internal patch representation (default: 15)
    int patchSize;
    /// number of ferns (simplified decision trees) in the ensemble classifier (default: 8)
    int numFerns;
    /// number of features (= tree depth) (default: 10)
    int featuresPerFern;
    /// sliding window scales = 1.2^n for n in {scaleMin, ... scaleMax} (default: -10, 11)
    int scaleMin, scaleMax;
    /// minimum width/height for sliding window (default 24)
    int bbMin;
    /// uses color histograms to extend the confidence measure of NNClassifier (experimental!)
    bool useColor;
    /// if set to true, "conservative confidence" is disabled (default: false)
    bool allowFastChange;
    /// temporary trains rotated patches to account for fast rotation in image plane (experimental!)
    bool enableFastRotation;

    /// Constructor setting default configuration
    MOTLDSettings(int cm = COLOR_MODE_GRAY)
    {
        colorMode = cm;
        patchSize = 15;
        numFerns = 8;
        featuresPerFern = 10;
        scaleMin = -10;
        scaleMax = 11;
        bbMin = 24;
        useColor = false;
        allowFastChange = false;
        enableFastRotation = false;
    }
};

/// The main class
class MultiObjectTLD
{
public:
    /** @brief Default constructor
     * @param width width...
     * @param height ...and height of the images in the sequence
     * @param settings configuration structure (see class MOTLDSettings for more details)
     */
    MultiObjectTLD(const int width, const int height, const MOTLDSettings &settings = MOTLDSettings())
        : ivWidth(width), ivHeight(height),
          ivColorMode(settings.colorMode), ivPatchSize(settings.patchSize), ivBBmin(settings.bbMin),
          ivUseColor(settings.useColor && ivColorMode == COLOR_MODE_RGB),
          ivEnableFastRotation(settings.enableFastRotation), ivLKTracker(LKTracker(width, height)),
          ivNNClassifier(NNClassifier(width, height, ivPatchSize, ivUseColor, settings.allowFastChange)),
          ivFernFilter(FernFilter(width, height, settings.numFerns, settings.featuresPerFern)),
          ivNObjects(0), ivLearningEnabled(true), ivNLastDetections(0) { };

    /** @brief Marks a new object in the previously passed frame.
     * @note To add multiple objects in a single frame please prefer addObjects().
     */
    void addObject(ObjectBox b);
    /** @brief Adds multiple objects in the previously passed frame.
     * @note For efficiency reasons all object boxes need to have the same aspect ratio. All
     *  further boxes are automatically reshaped to the aspect ratio of the first object box
     *  while preserving the area.
     */
    void addObjects(std::vector<ObjectBox> obs);
    /** @brief Processes the current frame (tracking - detecting - learning)
     * @param img The image passed as an unsigned char array. The pixels are assumed to be given
     *  row by row from top left to bottom right with the image width and height passed to the
     *  constructor (see MultiObjectTLD()). In case of COLOR_MODE_RGB the array should have the form
     *  [r_0, r_1, ..., r_n,  g_0, g_1, ..., g_n,  b_0, b_1, ..., b_n].
     */

    void deleteObject(int personID);

    void processFrame(unsigned char *img);
    /// En/Disables learning (i.e. updating the classifiers) at runtime
    void enableLearning(bool enable = true)
    {
        ivLearningEnabled = enable;
    };
    /** @brief Returns current status of object @c objId.
     * @returns one of the following states:
     *  @li @c STATUS_OK: the object is considered to be found correctly in the current frame
     *  @li @c STATUS_UNSURE: the object may be lost, but is still tracked
     *  @li @c STATUS_LOST: the object is definitely lost (the corresponding box returned by
     *    getObjectBoxes() is invalid)
     */
    int getStatus(const int objId = 0) const;
    /// Returns if the first object is valid (equivalent to @c getStatus(0) returning @c STATUS_OK)
    bool getValid() const
    {
        return ivNObjects > 0 ? ivValid[0] : false;
    };
    /// Returns the location of the first object.
    ObjectBox getObjectBox() const
    {
        return ivCurrentBoxes[0];
    };
    /// Returns the current object positions.
    std::vector<ObjectBox> getObjectBoxes() const
    {
        return ivCurrentBoxes;
    };

    std::vector<NNPatch> getObjectPatches() const
    {
        return ivCurrentPatches;
    };
    /** @brief Saves an output image to file in PPM format.
     * @param src the same as passed to processFrame()
     * @param filename the filename, suggested ending: ".ppm"
     * @param mode controls which components are drawn, bitwise or of DEBUG_DRAW_DETECTIONS,
     *  DEBUG_DRAW_PATCHES, DEBUG_DRAW_CROSSES.
     * @details The patches on the left side represent negative examples, positive examples are on
     *  the right side, starting a new column for each object. The red bars next to the patches
     *  sketch the corresponding color histograms.
     *  Green boxes represent detections shortlisted by the filtering process (see FernFilter).
     *  Blue boxes represent cluster means (if enabled).
     *  Blue crosses represent tracking points that are considered as inliers (see LKTracker).
     *  A red box stands for STATUS_OK, a yellow box means STATUS_UNSURE (cf. getStatus()).
     */
    void writeDebugImage(unsigned char *src, char *filename, int mode = 255) const;
    /// Writes a colored debug image into the given rgb matrices. Details see writeDebugImage().
    void getDebugImage(unsigned char *src, Matrix &rMat, Matrix &gMat, Matrix &bMat, int mode = 255) const;

    /// Returns an instance of MultiObjectTLD while loading the classifier from file.
    static MultiObjectTLD loadClassifier(const char *filename);
    /// Saves the classifier to a (binary) file.
    void saveClassifier(const char *filename) const;

private:
    int ivWidth;
    int ivHeight;
    //MOTLDSettings ivSettings;
    int ivColorMode;
    int ivPatchSize;
    int ivBBmin;
    bool ivUseColor;
    bool ivEnableFastRotation;
    LKTracker ivLKTracker;
    NNClassifier ivNNClassifier;
    FernFilter ivFernFilter;

    int ivNObjects;
    float ivAspectRatio;
    std::vector<ObjectBox> ivCurrentBoxes;
    std::vector<bool> ivDefined;
    std::vector<bool> ivValid;
    std::vector<NNPatch> ivCurrentPatches;
    bool ivLearningEnabled;

    Matrix ivCurImage;
    unsigned char *ivCurImagePtr;
    std::vector<FernDetection> ivLastDetections;
    std::vector<FernDetection> ivLastDetectionClusters;
    int ivNLastDetections;
    void clusterDetections(float threshold);

    MultiObjectTLD (int width, int height, int colorMode, int patchSize, int bbMin, bool useColor,
                    bool fastRotation, NNClassifier nnc, FernFilter ff, int nObjects,
                    float aspectRatio, bool learningEnabled);
};


#endif // MULTIOBJECTTLD_H
