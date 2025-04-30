#include <iostream>
#include <opencv2/opencv.hpp>

#include <unordered_set>

// Debug
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

ofstream logFile("log.txt");

vector<double> applyGaussianWeightAverage(int windowSize, double sigma)
{
	vector<double> weights(windowSize);
	double sum = 0.0;

	// Ensure the window is centered
	int center = windowSize / 2;

	for (int i = 0; i < windowSize; ++i)
	{
		int distance = i - center;
		weights[i] = exp(-(distance * distance) / (2.0 * sigma * sigma));
		sum += weights[i];
	}

	// Normalize so the total sum of weights is 1.0
	for (double &w : weights)
	{
		w /= sum;
	}

	return weights;
}

int main(int argc, char **argv)
{
	// Validation; Argument should have two parameters.
	if (argc != 2)
	{
		cerr << "[main] [Error]: " << argv[0] << " <input image file name> e.g. ./src/main 2DEmpty.jpg" << endl;
		return -1;
	}

	try
	{

		VideoCapture cap;

		if (argc != 2)
		{
			cerr << "[main] [Error]: " << argv[0] << " <input video file name> e.g. ./src/main shaky.mp4" << endl;
			return -1;
		}

		cap.open(argv[1]);
		if (!cap.isOpened())
		{
			cerr << "[main] [Error]: Could not open the video file: " << argv[1] << endl;
			return -1;
		}

		const int windowSize = 19;
		const double sigma = 5.0; // Fixed typo in variable name

		deque<Mat> frameBuffer;
		deque<Mat> matrixBuffer;

		vector<double> gaussianWeight = applyGaussianWeightAverage(windowSize, sigma); // Fixed vector typea);

		logFile << "[main] [Debug]: Gaussian weights: ";
		for (const auto &weight : gaussianWeight)
		{
			logFile << weight << " ";
		}
		logFile << endl;

		Mat frame, stableFrame;
		namedWindow("Original Video", WINDOW_NORMAL);
		namedWindow("Stable Video", WINDOW_NORMAL);

		while (true)
		{
			// Read the next frame from the video file
			cap >> frame;
			if (frame.empty())
			{
				break;
			}

			frameBuffer.push_back(frame.clone());

			// Keep buffer at specified window size
			if (frameBuffer.size() > windowSize)
			{
				frameBuffer.pop_front();
			}

			// Display the original frame
			imshow("Original Video", frame);

			// Only show the stable frame if we have a valid frame
			stableFrame = frame.clone(); // For now, just use original frame
			imshow("Stable Video", stableFrame);

			if (waitKey(30) >= 0) // Wait for 30ms or until a key is pressed
			{
				break;
			}
		}

		cout << "[main] [Debug] Successfully processed" << endl;

		// Debug
		logFile.close();

		return 0;
	}
	catch (const std::invalid_argument &e)
	{
		cerr << "[main] [Error]: Invalid argument Exception: " << e.what() << "\n";
		return -1;
	}
	catch (const cv::Exception &e)
	{
		cerr << "[main] [Error]: OpenCV Exception: " << e.what() << endl;
		return -1;
	}
	catch (const std::exception &e)
	{
		cerr << "[main] Error: Standard Exception: " << e.what() << endl;
		return -1;
	}
	catch (...)
	{
		cerr << "[main] [Error]: An unknown error occurred" << endl;
		return -1;
	}
}