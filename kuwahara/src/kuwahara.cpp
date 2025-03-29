#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>

// Debug
// #include <iostream>
// #include <fstream>

using namespace cv;
using namespace std;
using namespace chrono;

/// @brief Struct to hold the coordinates of a quadrant.
struct Quadrant
{
	int y1, x1, y2, x2;
};

/// @brief Struct to hold the mean and variance of a region.
struct RegionStatistics
{
	double mean = 0.0;
	double variance = DBL_MAX;
};

/// @brief Calculate the mean and variance of a region.
/// @details The region is defined by the top-left and bottom-right corners.
/// The mean and variance are calculated using the integral images.
/// The mean is calculated as:
/// mean = sum / numberOfPixels
/// The variance is calculated as:
/// variance = (sqSum / numberOfPixels) - (mean * mean)
/// @param sumImage Sum image
/// @param sqSumImage  square sum image
/// @param quad quadrant
/// @return RegionStatistics
RegionStatistics calculateRegionStatistics(const Mat &sumImage, const Mat &sqSumImage, const Quadrant &quad)
{
	RegionStatistics stats;

	if (quad.y1 >= quad.y2 || quad.x1 >= quad.x2)
	{
		return stats;
	}

	int numberOfPixels = (quad.y2 - quad.y1 + 1) * (quad.x2 - quad.x1 + 1);

	double sum = sumImage.at<double>(quad.y2 + 1, quad.x2 + 1) -
							 sumImage.at<double>(quad.y1, quad.x2 + 1) -
							 sumImage.at<double>(quad.y2 + 1, quad.x1) +
							 sumImage.at<double>(quad.y1, quad.x1);

	double sqSum = sqSumImage.at<double>(quad.y2 + 1, quad.x2 + 1) -
								 sqSumImage.at<double>(quad.y1, quad.x2 + 1) -
								 sqSumImage.at<double>(quad.y2 + 1, quad.x1) +
								 sqSumImage.at<double>(quad.y1, quad.x1);

	stats.mean = sum / numberOfPixels;
	stats.variance = (sqSum / numberOfPixels) - (stats.mean * stats.mean);

	return stats;
}

/// @brief Generate mock image with 5x5
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

/// @brief Get quadrants of the image.
/// @details The quadrants are defined as follows:
/// A: Top-left, B: Top-right, C: Bottom-left, D: Bottom-right
/// @param x column index
/// @param y row index
/// @param halfKernel half of the kernel size
/// @param rows rows of the image
/// @param cols columns of the image
/// @return vector of quadrants
vector<Quadrant> getQuadrants(int x, int y, int halfKernel, int rows, int cols)
{
	return {
			// A
			{max(0, y - halfKernel), max(0, x - halfKernel), y, x},

			// B
			{max(0, y - halfKernel), x, y, min(cols - 1, x + halfKernel)},

			// C
			{y, max(0, x - halfKernel), min(rows - 1, y + halfKernel), x},

			// D
			{y, x, min(rows - 1, y + halfKernel), min(cols - 1, x + halfKernel)}};
}

/// @brief Calculate Integral Image and Sqaure Integraal Image. (Summed Area Tables (SAT))
/// @details SAT tables will be used to calculate the mean and variances of regions in Kuwahara Filter
/// The details of the algorithm can be found in https://en.wikipedia.org/wiki/Summed-area_table#:~:text=In%20the%20image%20processing%20domain,Crow%20for%20use%20with%20mipmaps.
/// e.g arguments - ./src/kuwahara limes.tif output1.jpg 5
/// e.g. arguments for debugging - ./src/kuwahara debug output1.jpg 5
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

/// @brief This works by four big four steps
/// 1. Calculate the integral image and square integral image. Still Big O(n) time compexity.
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
			auto quadrants = getQuadrants(x, y, halfKernelSize, input.rows, input.cols);

			// Debug: Log each region's values to the log.txt file for debugging
			// for (int i = 0; i < 4; i++)
			// {
			// 	logFile << "Region " << i + 1 << ": ";
			// 	logFile << "Row " << quadrants[i][0] << " to " << quadrants[i][2] << ", ";
			// 	logFile << "Col " << quadrants[i][1] << " to " << quadrants[i][3] << std::endl;
			// }

			// Calculate mean and variance for each region
			double minVariance = DBL_MAX;
			double meanWithSmallestVariance = 0.0;

			// Loop through quadrants
			for (const Quadrant &quad : quadrants)
			{
				RegionStatistics stats = calculateRegionStatistics(sumImage, sqSumImage, quad);

				if (stats.variance < minVariance)
				{
					minVariance = stats.variance;
					meanWithSmallestVariance = stats.mean;
				}
			}

			// Set output pixel to the mean of the region with the smallest variance
			output.at<uchar>(y, x) = saturate_cast<uchar>(meanWithSmallestVariance);
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
		cerr << "Invalid argument Exception: " << e.what() << "\n";
		return -1;
	}
	catch (const cv::Exception &e)
	{
		cerr << "OpenCV Exception: " << e.what() << std::endl;
		return -1;
	}
	catch (const std::exception &e)
	{
		cerr << "Standard Exception: " << e.what() << std::endl;
		return -1;
	}
	catch (...)
	{
		cerr << "An unknown error occurred" << std::endl;
		return -1;
	}
}