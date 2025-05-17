#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

ofstream logFile("log.txt");

const int grassLabel = 0;
const int cloudLabel = 1;
const int seaLabel = 2;
const int K = 3; // kNN parameter

// Texture Analysis
// 		IMPLEMENT AND TRAIN A SIMPLE APPROACH TO CLASSIFY TEXTURE WITHIN IMAGES.USING THE SIMPLE CLASSIFIER,
// 		SEGMENT GRASS, CLOUDS AND SEA FROM IMAGES.

// Compute LBP value for a single pixel
uchar calculateLBP(const Mat &img, int x, int y)
{
	uchar center = img.at<uchar>(y, x);
	uchar code = 0;

	// Neighbors positions (clockwise starting from top-left)
	uchar topLeft = img.at<uchar>(y - 1, x - 1);
	uchar top = img.at<uchar>(y - 1, x);
	uchar topRight = img.at<uchar>(y - 1, x + 1);
	uchar right = img.at<uchar>(y, x + 1);
	uchar bottomRight = img.at<uchar>(y + 1, x + 1);
	uchar bottom = img.at<uchar>(y + 1, x);
	uchar bottomLeft = img.at<uchar>(y + 1, x - 1);
	uchar left = img.at<uchar>(y, x - 1);

	// Build the LBP code by comparing neighbors with the center
	code |= (topLeft > center) << 7;
	code |= (top > center) << 6;
	code |= (topRight > center) << 5;
	code |= (right > center) << 4;
	code |= (bottomRight > center) << 3;
	code |= (bottom > center) << 2;
	code |= (bottomLeft > center) << 1;
	code |= (left > center) << 0;

	return code;
}

// Compute LBP histogram (256-bin) for a patch
Mat computeLBPHistogram(const Mat &patch)
{
	Mat hist = Mat::zeros(1, 256, CV_32F);
	// Iterate over the patch and compute LBP values
	for (int y = 1; y < patch.rows - 1; ++y)
	{
		for (int x = 1; x < patch.cols - 1; ++x)
		{
			uchar lbp = calculateLBP(patch, x, y);
			// Leave the row as 0, we are only incrementing the column histogram.
			// E.g. if 216 is the LBP value, increment the 216th column of the histgram. hist[216] +=1.
			hist.at<float>(0, lbp)++;
		}
	}

	// Normalize the histogram so that the sum of all bins equals 1
	normalize(hist, hist, 1.0, 0.0, NORM_L1);
	return hist;
}

// Loads training images from a specified folder, extracts LBP histogram features, and stores them with corresponding labels.
void loadTrainingData(const string &folder, int label, Mat &features, Mat &labels)
{
	// In each folder, loop the file
	for (const auto &entry : std::filesystem::directory_iterator(folder))
	{
		// Grayscale image should be used for as LBP doens't care about the colour but texture
		Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);
		if (img.empty())
		{
			logFile << "Failed to load image: " << entry.path() << endl;
			continue;
		}

		Mat feature;
		resize(img, feature, Size(64, 64)); // Resize to a fixed size
		Mat hist = computeLBPHistogram(img);
		features.push_back(hist);
		labels.push_back(label);
	}
}

int main(int argc, char **argv)
{
	try
	{
		if (argc != 2)
		{
			cerr << "Usage: " << argv[0] << " <path_to_test_image>" << endl;
			return -1;
		}

		Mat features, labels;

		// Prepare training data
		loadTrainingData("../data/grass", grassLabel, features, labels);
		loadTrainingData("../data/cloud", cloudLabel, features, labels);
		loadTrainingData("../data/sea", seaLabel, features, labels);

		logFile << "Training data loaded successfully." << endl;
		cout << "Training data loaded successfully." << endl;

		// Train Classifier
		Ptr<KNearest> knn = KNearest::create();
		knn->setDefaultK(K);
		knn->train(features, ROW_SAMPLE, labels);

		logFile << "Classifier trained successfully." << endl;
		cout << "Classifier trained successfully." << endl;

		// Load test image using command line argument
		string fullInputPath = "../data/" + string(argv[1]);
		Mat testImg = imread(fullInputPath, IMREAD_GRAYSCALE);

		if (testImg.empty())
		{
			logFile << "Failed to load test image: " << argv[1] << endl;
			cerr << "Failed to load test image: " << argv[1] << endl;
			return -1;
		}
		logFile << "Test image loaded successfully: " << argv[1] << endl;
		cout << "Test image loaded successfully: " << argv[1] << endl;

		// Compute LBP histogram for the test image
		Mat result = Mat::zeros(testImg.size(), CV_8UC3);
		int patchSize = 32;

		for (int y = 0; y < testImg.rows - patchSize; y += patchSize)
		{
			for (int x = 0; x < testImg.cols - patchSize; x += patchSize)
			{
				Mat patch = testImg(Rect(x, y, patchSize, patchSize));
				Mat hist = computeLBPHistogram(patch);

				// Distance-weighted kNN
				Mat neighborResponses, neighborDistances;
				Mat results; // Create a Mat to store the results
				knn->findNearest(hist, K, results, neighborResponses, neighborDistances);

				float weightedVotes[3] = {0};
				for (int i = 0; i < K; ++i)
				{
					float response = neighborResponses.at<float>(0, i);
					float distance = neighborDistances.at<float>(0, i);
					float weight = (distance == 0) ? FLT_MAX : 1.0f / distance;

					if (response == grassLabel)
						weightedVotes[0] += weight;
					else if (response == cloudLabel)
						weightedVotes[1] += weight;
					else if (response == seaLabel)
						weightedVotes[2] += weight;
				}

				int finalClass = max_element(weightedVotes, weightedVotes + 3) - weightedVotes;

				Scalar color;
				if (finalClass == grassLabel)
					color = Scalar(0, 255, 0); // Grass as green
				else if (finalClass == cloudLabel)
					color = Scalar(200, 200, 200); // Clouds as grey
				else
					color = Scalar(255, 0, 0); // Sea as blue

				rectangle(result, Rect(x, y, patchSize, patchSize), color, FILLED);
			}
		}

		// Display & save the result
		imshow("Segmented Texture", result);
		imshow("Original Image", testImg);
		waitKey(0);

		logFile << "Segmentation completed successfully." << endl;
		cout << "Segmentation completed successfully." << endl;

		logFile.close();
	}
	catch (const std::exception &e)
	{
		logFile << "Exception: " << e.what() << endl;
		cerr << "Exception: " << e.what() << endl;
		logFile.close();
		return -1;
	}

	return 0;
}