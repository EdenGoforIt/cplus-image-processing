#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>

// For Logs
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;
using namespace chrono;

/// @brief Generate mock image with 5x5 for testing
/// @return 5x5 image with 32-bit signed integers
Mat generate5x5MatImage()
{
	cv::Mat image(5, 5, CV_8U);
	int counter = 0;

	for (int row = 0; row < image.rows; ++row)
	{
		for (int col = 0; col < image.cols; ++col)
		{
			image.at<int>(row, col) = counter++;
		}
	}

	return image;
}

/// @brief Calculate Integral Image and Sqaure Integraal Image. (Summed Area Tables (SAT))
/// @details SAT tables will be used to calculate the mean and variances of regions in Kuwahara Filter
/// The details of the algorithm can be found in https://en.wikipedia.org/wiki/Summed-area_table#:~:text=In%20the%20image%20processing%20domain,Crow%20for%20use%20with%20mipmaps.
/// e.g arguments - .src/kuwahara limes.tif output1.jpg 5
/// e.g. arguments for debugging - .src/kuwahara debug output1.jpg 5
/// @param input input Image to calculate Sum and Square Sum images
/// @param sumImage output Sum Image
/// @param sqSumImage output Square Sum Image
void calculateIntegralImages(const Mat &input, Mat &sumImage, Mat &sqSumImage)
{
	// When creating the integral image, it is convenient to create a matrix (or IplImage)
	// with one extra column and one extra row, both initialised to zero. This is to avoid when x or y becomes -1
	sumImage = Mat::zeros(input.rows + 1, input.cols + 1, CV_64F);
	sqSumImage = Mat::zeros(input.rows + 1, input.cols + 1, CV_64F);

	// Fill integral images row by row. Iterate the height (rows) first
	for (int y = 0; y < input.rows; y++)
	{
		// Iterate the width (cols) of the image
		for (int x = 0; x < input.cols; x++)
		{
			// Get pixel value (as double for precision)
			double pixel = input.at<double>(y, x);
			double pixelSq = pixel * pixel;

			// Calculate sums (+1 because integral images have extra row/col)
			// I(x,y) = i(x,y) + I(x-1, y) + I(x,y -1) +  - I(x-1, y-1). In Matrix, element is accessed by matrix[row, col] = matrix [y,x]
			sumImage.at<double>(y + 1, x + 1) = pixel +
																					sumImage.at<double>(y, x + 1) +
																					sumImage.at<double>(y + 1, x) -
																					sumImage.at<double>(y, x);

			// Sqaure Sum will be used to calculate variance
			sqSumImage.at<double>(y + 1, x + 1) = pixelSq +
																						sqSumImage.at<double>(y, x + 1) +
																						sqSumImage.at<double>(y + 1, x) -
																						sqSumImage.at<double>(y, x);
		}
	}
}

/// @brief This works by four big steps
/// 1. Calculate the integral image and square integral image
/// 2. Calculate the mean and variance of each region
/// 3. Find the region with the smallest variance
/// 4. Set the output pixel to the mean of the region with the smallest variance
/// @param input Input image
/// @param output Output image
/// @param kernelSize Kernel size
void kuwaharaFilter(const Mat &input, Mat &output, int kernelSize)
{
	// Debug
	// std::ofstream logFile("log.txt");

	// Create output image
	output = Mat(input.size(), input.type());
	int halfKernelSize = kernelSize / 2;

	// Calculate integral images before to be used without complexiing the code
	// According to the book, it might be safer to convert to double. Or we can calculate the max by max = width*height*255
	Mat sumImage, sqSumImage;
	Mat inputDoubleImage;
	input.convertTo(inputDoubleImage, CV_64F);

	// Calculate integral images already to improve performance
	calculateIntegralImages(inputDoubleImage, sumImage, sqSumImage);

	// Rows
	for (int y = 0; y < input.rows; y++)
	{
		// Then columns
		for (int x = 0; x < input.cols; x++)
		{
			// Arrays to store region info
			double means[4] = {0};
			double variances[4] = {0};

			// Define the 4 regions (quadrants) around current pixel
			//  A | B
			//  C | D

			// Region 1: Top-Left (A)
			int a_y1 = max(0, y - halfKernelSize);
			int a_y2 = y;
			int a_x1 = max(0, x - halfKernelSize);
			int a_x2 = x;

			// Region 2: Top-Right (B)
			int b_y1 = max(0, y - halfKernelSize);
			int b_y2 = y;
			int b_x1 = x;
			int b_x2 = min(input.cols - 1, x + halfKernelSize);

			// Region 3: Bottom-Left (C)
			int c_y1 = y;
			int c_y2 = min(input.rows - 1, y + halfKernelSize);
			int c_x1 = max(0, x - halfKernelSize);
			int c_x2 = x;

			// Region 4: Bottom-Right (D)
			int d_y1 = y;
			int d_y2 = min(input.rows - 1, y + halfKernelSize);
			int d_x1 = x;
			int d_x2 = min(input.cols - 1, x + halfKernelSize);

			// Quadrants coordinates array
			int quadrants[4][4] = {
					{a_y1, a_x1, a_y2, a_x2},
					{b_y1, b_x1, b_y2, b_x2},
					{c_y1, c_x1, c_y2, c_x2},
					{d_y1, d_x1, d_y2, d_x2}};

			// Debug: Log each region's values to the log.txt file for debugging
			// for (int i = 0; i < 4; i++)
			// {
			// 	logFile << "Region " << i + 1 << ": ";
			// 	logFile << "Row " << quadrants[i][0] << " to " << quadrants[i][2] << ", ";
			// 	logFile << "Col " << quadrants[i][1] << " to " << quadrants[i][3] << std::endl;
			// }

			// Calculate mean and variance for each region
			double minVariance = DBL_MAX;
			int lowestVarianceIndex = 0;

			// Loop through quadrants
			for (int i = 0; i < 4; i++)
			{
				int y1 = quadrants[i][0];
				int x1 = quadrants[i][1];
				int y2 = quadrants[i][2];
				int x2 = quadrants[i][3];

				// Edge protection
				if (y1 >= y2 || x1 >= x2)
				{
					continue;
				}

				int numberOfPixels = (y2 - y1 + 1) * (x2 - x1 + 1);

				// Calculate sum and square sum of regions. pt4 - pt1 - pt2 + pt3
				// + 1 as Integral Images have padding to avoid negative indices
				double sum = sumImage.at<double>(y2 + 1, x2 + 1) -
										 sumImage.at<double>(y1, x2 + 1) -
										 sumImage.at<double>(y2 + 1, x1) +
										 sumImage.at<double>(y1, x1);

				double sqSum = sqSumImage.at<double>(y2 + 1, x2 + 1) -
											 sqSumImage.at<double>(y1, x2 + 1) -
											 sqSumImage.at<double>(y2 + 1, x1) +
											 sqSumImage.at<double>(y1, x1);

				// Debug
				// logFile << "Sum: " << sum << " | Square Sum:  " << sqSum << endl;

				// Calculate mean and variance
				means[i] = sum / numberOfPixels;
				variances[i] = (sqSum / numberOfPixels) - (means[i] * means[i]);

				// Debug
				// logFile << "Min: " << means[i] << " | Square Sum:  " << variances[i] << endl;

				// Keep track of the minimum variacne
				if (variances[i] < minVariance)
				{
					minVariance = variances[i];
					lowestVarianceIndex = i;

					// Debug
					// logFile << "Found min variance: " << minVariance << endl;
				}
			}

			// Set output pixel to the mean of the region with the smallest variance
			output.at<uchar>(y, x) = saturate_cast<uchar>(means[lowestVarianceIndex]);
		}
	}

	// Debug
	// logFile.close();
}

int main(int argc, char **argv)
{
	// Argument Validation. Argument should have three parameters.
	if (argc != 4)
	{
		cerr << "!! Wrong Arguments. " << argv[0] << " <input> <output> <kernel_size>. e.g. ./src/kuwahara limes_kuwahara5x5.tif output1.jpg 5" << endl;
		return -1;
	}

	try
	{
		// Parse Arguments
		const char *inputPath = argv[1];
		const char *outputPath = argv[2];
		int kernelSize = atoi(argv[3]);

		cout << "Arguments - Input: " << inputPath << ". Output: " << outputPath << ". Kernel Size: " << kernelSize << endl;

		// Kernel Size Validation. Odd nubmer works better.
		if (kernelSize % 2 == 0 || kernelSize < 3 || kernelSize > 15)
		{
			cerr << "!! Kernel Size should be odd and between 3 and 15." << endl;
			return -1;
		}

		// Read Image. The image should be under the root directory. Otherwise change the path below.
		Mat inputImage;
		if (strcmp(inputPath, "debug") == 0)
		{
			inputImage = generate5x5MatImage();
		}
		else
		{
			string fullInputPath = "../" + string(inputPath);
			inputImage = imread(fullInputPath, IMREAD_GRAYSCALE);

			if (inputImage.empty())
			{
				cerr << "!! Input Image cannot be read." << endl;
				return -1;
			}
		}

		// Timer starts
		auto startTime = high_resolution_clock::now();

		Mat outputImage;
		kuwaharaFilter(inputImage, outputImage, kernelSize);

		// Measure time
		auto stopTime = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stopTime - startTime);
		cout << "Processing time: " << duration.count() / 1000.0 << " milliseconds" << endl;

		// The image should be under the root directory. Otherwise change the path below.
		string fullOutputPath = "../" + string(outputPath);
		imwrite(fullOutputPath, outputImage);
		cout << "Successfully applied Kuwahara Filter." << endl;
		return 0;
	}
	catch (const std::invalid_argument &e)
	{
		std::cerr << "Invalid argument Exception: " << e.what() << "\n";
		return -1;
	}
	catch (const cv::Exception &e)
	{
		std::cerr << "OpenCV Exception: " << e.what() << std::endl;
		return -1;
	}
	catch (const std::exception &e)
	{
		std::cerr << "Standard Exception: " << e.what() << std::endl;
		return -1;
	}
	catch (...)
	{
		std::cerr << "An unknown error occurred" << std::endl;
		return -1;
	}
}