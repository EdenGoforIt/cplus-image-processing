#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>

using namespace cv;
using namespace cv::ml;
using namespace std;
namespace fs = std::filesystem;

const int grassLabel = 0;
const int cloudLabel = 1;
const int seaLabel = 2;
const int K = 3; // kNN parameter

int confusionMatrix[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

uchar calculateLBP(const Mat &img, int x, int y)
{
	uchar center = img.at<uchar>(y, x);
	uchar code = 0;

	code |= (img.at<uchar>(y - 1, x - 1) > center) << 7;
	code |= (img.at<uchar>(y - 1, x) > center) << 6;
	code |= (img.at<uchar>(y - 1, x + 1) > center) << 5;
	code |= (img.at<uchar>(y, x + 1) > center) << 4;
	code |= (img.at<uchar>(y + 1, x + 1) > center) << 3;
	code |= (img.at<uchar>(y + 1, x) > center) << 2;
	code |= (img.at<uchar>(y + 1, x - 1) > center) << 1;
	code |= (img.at<uchar>(y, x - 1) > center) << 0;

	return code;
}

Mat computeLBPHistogram(const Mat &patch)
{
	Mat hist = Mat::zeros(1, 256, CV_32F);
	for (int y = 1; y < patch.rows - 1; ++y)
	{
		for (int x = 1; x < patch.cols - 1; ++x)
		{
			uchar lbp = calculateLBP(patch, x, y);
			hist.at<float>(0, lbp)++;
		}
	}
	normalize(hist, hist, 1.0, 0.0, NORM_L1);
	return hist;
}

void printConfusionMatrix()
{
	cout << "\n+-----------+-------+-------+-------+\n";
	cout << "| Predicted | Grass | Cloud | Sea   |\n";
	cout << "+-----------+-------+-------+-------+\n";
	string classes[3] = {"Grass    ", "Cloud    ", "Sea      "};
	for (int i = 0; i < 3; ++i)
	{
		cout << "| " << classes[i] << "|";
		for (int j = 0; j < 3; ++j)
		{
			cout << "   " << confusionMatrix[i][j] << "  |";
		}
		cout << endl;
	}
	cout << "+-----------+-------+-------+-------+\n"
			 << endl;
}

void testPatches(const string &testFolder, int trueLabel, Ptr<KNearest> &knn)
{
	for (const auto &entry : fs::directory_iterator(testFolder))
	{
		Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);
		if (img.empty())
			continue;

		Mat hist = computeLBPHistogram(img);
		Mat neighborResponses, neighborDistances, results;
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
		confusionMatrix[trueLabel][finalClass]++;
	}
}

int main()
{
	// Load pre-trained model directly (retraining not shown in this script)
	Ptr<KNearest> knn = KNearest::create();
	// Assume you’ve already trained and loaded data into 'knn' somewhere before calling this test.

	// Test on each directory
	testPatches("../test/grass", grassLabel, knn);
	testPatches("../test/cloud", cloudLabel, knn);
	testPatches("../test/sea", seaLabel, knn);

	printConfusionMatrix();

	return 0;
}