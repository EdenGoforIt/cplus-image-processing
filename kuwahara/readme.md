# ASSIGNMENT 1: KUWAHARA FILTER IMPLEMENTATION

## OVERVIEW

This assignment implements the Kuwahara Filter for image processing, utilizing Integral Images (Summed-Area Tables) to enhance performance. The build process may vary across Linux, macOS, and Windows. This program was executed on a Mac M2, which operates differently from other systems, though the fundamentals of how to execute the programs remain the similar.

## REQUIREMENTS

- Open CV 4++

## BUILDING THE PROJECT USING CMAKE

1. Build with CMake

```
   $ mkdir -p build
   $ cd build
   $ cmake ..
   $ make
```

2. Execute the file in the `build` folder with Arguments

```
$ .src/main input.jpg output.jpg 5
```

Parameters:

1. input.jpg - Input image file (will be converted to grayscale)
2. output.jpg - Output image file
3. 7 - Neighborhood size (odd number between 3 and 15+)

## IMPLEMENTATION DETAILS

Kuwahara Filter Algorithm
The Kuwahara filter works by:

1. Dividing the neighborhood around each pixel into 4 square regions
2. Computing the mean and variance for each region
3. Selecting the region with the minimum variance
4. Replacing the pixel with the mean of the selected region

Performance Optimization
This implementation uses Summed-Area Tables (SAT) for efficient calculation:

- One SAT for the sum of pixel values
- One SAT for the sum of squared pixel values
- Calculation of mean and variance using O(1) operations instead of O(n²)

## TESTING

Test the implementation with different images and kernel sizes:

```
$ ./src/kuwahara limes output1.jpg 5
```

For debugging use the following

```
./src/kuwahara debug output1.jpg 5
```

CMAKE CONFIGURATION
The project uses the following CMake configuration:

cmake_minimum_required(VERSION 3.11)
project(KuwaharaFilter)

## References

To gain a deeper understanding of the algorithm, this program references the following sources:

- (Wikipedia)[https://en.wikipedia.org/wiki/Kuwahara_filter]
- (What is a Kuwahara Filter)[https://medium.com/swlh/what-is-a-kuwahara-filter-77921ce286f2]
- (An Image Denoising Algorithm based on Kuwahara Filter)[https://www.arivis.com/blog/denoising-images-using-kuwahara-filter]
  -(Image Fusion Using Kuwahara Filter; ISSN: 0975-9646)[https://www.ijcsit.com/docs/Volume%205/vol5issue04/ijcsit20140504242.pdf]
- (Adaptive Kuwahara ﬁlter)[https://www.researchgate.net/publication/282512229_Adaptive_Kuwahara_filter]
