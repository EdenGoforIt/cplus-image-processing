#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <fstream>
#include <limits> // Required for DBL_MAX

using namespace cv;
using namespace std;
using namespace chrono;

Mat generate5x5MatImage()
{
	cv::Mat image(5, 5, CV_32SC1);
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

void calculateIntegralImages(const Mat &input, Mat &sumImage, Mat &sqSumImage)
{
	sumImage = Mat::zeros(input.rows + 1, input.cols + 1, CV_64F);
	sqSumImage = Mat::zeros(input.rows + 1, input.cols + 1, CV_64F);

	for (int y = 0; y < input.rows; y++)
	{
		for (int x = 0; x < input.cols; x++)
		{
			double pixel;
			if (input.type() == CV_8U)
			{
				pixel = input.at<uchar>(y, x);
			}
			else if (input.type() == CV_32SC1)
			{
				pixel = static_cast<double>(input.at<int>(y, x));
			}
			else
			{
				throw std::runtime_error("Unsupported input image type for calculateIntegralImages");
			}
			double pixelSq = pixel * pixel;

			sumImage.at<double>(y + 1, x + 1) = pixel +
																					sumImage.at<double>(y, x + 1) +
																					sumImage.at<double>(y + 1, x) -
																					sumImage.at<double>(y, x);

			sqSumImage.at<double>(y + 1, x + 1) = pixelSq +
																						sqSumImage.at<double>(y, x + 1) +
																						sqSumImage.at<double>(y + 1, x) -
																						sqSumImage.at<double>(y, x);
		}
	}
}

void kuwaharaFilter(const Mat &input, Mat &output, int kernelSize)
{
	std::ofstream logFile("log.txt");

	output = Mat(input.size(), input.type());
	int halfKernelSize = kernelSize / 2;

	Mat sumImage, sqSumImage;
	Mat inputDoubleImage;
	input.convertTo(inputDoubleImage, CV_64F);

	calculateIntegralImages(inputDoubleImage, sumImage, sqSumImage);

	for (int y = 0; y < input.rows; y++)
	{
		for (int x = 0; x < input.cols; x++)
		{
			double regionMean[4] = {0};
			double regionVariance[4] = {0};
			int regionSize[4] = {0};

			int r1_y1 = max(0, y - halfKernelSize);
			int r1_y2 = y;
			int r1_x1 = max(0, x - halfKernelSize);
			int r1_x2 = x;

			int r2_y1 = max(0, y - halfKernelSize);
			int r2_y2 = y;
			int r2_x1 = x;
			int r2_x2 = min(input.cols, x + halfKernelSize + 1); // Corrected upper bound

			int r3_y1 = y;
			int r3_y2 = min(input.rows, y + halfKernelSize + 1); // Corrected upper bound
			int r3_x1 = max(0, x - halfKernelSize);
			int r3_x2 = x;

			int r4_y1 = y;
			int r4_y2 = min(input.rows, y + halfKernelSize + 1); // Corrected upper bound
			int r4_x1 = x;
			int r4_x2 = min(input.cols, x + halfKernelSize + 1); // Corrected upper bound

			regionSize[0] = (r1_y2 - r1_y1 + 1) * (r1_x2 - r1_x1 + 1);
			regionSize[1] = (r2_y2 - r2_y1 + 1) * (r2_x2 - r2_x1); // Corrected size calculation
			regionSize[2] = (r3_y2 - r3_y1) * (r3_x2 - r3_x1 + 1); // Corrected size calculation
			regionSize[3] = (r4_y2 - r4_y1) * (r4_x2 - r4_x1);		 // Corrected size calculation

			int regions[4][4] = {
					{r1_y1, r1_x1, r1_y2, r1_x2},
					{r2_y1, r2_x1, r2_y2, r2_x2},
					{r3_y1, r3_x1, r3_y2, r3_x2},
					{r4_y1, r4_x1, r4_y2, r4_x2}};

			for (int i = 0; i < 4; i++)
			{
				logFile << "Region " << i + 1 << ": ";
				logFile << "Row " << regions[i][0] << " to " << regions[i][2] << ", ";
				logFile << "Col " << regions[i][1] << " to " << regions[i][3] << std::endl;
			}

			double minVariance = std::numeric_limits<double>::max();
			int minVarIndex = 0;

			for (int i = 0; i < 4; i++)
			{
				if (regionSize[i] > 0)
				{
					int y1 = regions[i][0];
					int x1 = regions[i][1];
					int y2 = regions[i][2];
					int x2 = regions[i][3];

					double sum = sumImage.at<double>(y2 + 1, x2 + 1) -
											 sumImage.at<double>(y1, x2 + 1) -
											 sumImage.at<double>(y2 + 1, x1) +
											 sumImage.at<double>(y1, x1);

					double sqSum = sqSumImage.at<double>(y2 + 1, x2 + 1) -
												 sqSumImage.at<double>(y1, x2 + 1) -
												 sqSumImage.at<double>(y2 + 1, x1) +
												 sqSumImage.at<double>(y1, x1);

					regionMean[i] = sum / regionSize[i];
					regionVariance[i] = (sqSum / regionSize[i]) - (regionMean[i] * regionMean[i]);

					if (regionVariance[i] < minVariance)
					{
						minVariance = regionVariance[i];
						minVarIndex = i;
					}
				}
			}
			output.at<uchar>(y, x) = saturate_cast<uchar>(regionMean[minVarIndex]);
		}
	}
	logFile.close();
}

int main(int argc, char **argv)
{
	if (argc != 4)
	{
		cerr << "!! Wrong Arguments. " << argv[0] << " <input> <output> <kernel_size>. e.g. ./src/kuwahara limes_kuwahara5x5.tif output1.jpg 5" << endl;
		return -1;
	}

	try
	{
		const char *inputPath = argv[1];
		const char *outputPath = argv[2];
		int kernelSize = atoi(argv[3]);

		cout << "Converted - Input: " << inputPath << ". Output: " << outputPath << ". Kernel Size: " << kernelSize << endl;

		if (kernelSize % 2 == 0 || kernelSize < 3 || kernelSize > 15)
		{
			cerr << "!! Kernel Size should be odd and between 3 and 15." << endl;
			return -1;
		}

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

		auto startTime = high_resolution_clock::now();

		Mat outputImage;
		kuwaharaFilter(inputImage, outputImage, kernelSize);

		auto stopTime = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stopTime - startTime);
		cout << "Processing time: " << duration.count() / 1000.0 << " milliseconds" << endl;

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
		return 1;
	}
	catch (const std::exception &e)
	{
		std::cerr << "Standard Exception: " << e.what() << std::endl;
		return 1;
	}
	catch (...)
	{
		std::cerr << "An unknown error occurred" << std::endl;
		return 1;
	}
}