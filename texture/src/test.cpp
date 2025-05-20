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
	// Calculate total files and correct predictions for each class
	int totalByClass[3] = {0, 0, 0};
	int correctByClass[3] = {0, 0, 0};

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			totalByClass[i] += confusionMatrix[i][j];
			if (i == j)
			{
				correctByClass[i] = confusionMatrix[i][j];
			}
		}
	}

	// Print header
	cout << "\n=== CLASSIFICATION REPORT ===\n\n";

	// Print accuracy per class
	string classNames[3] = {"Grass", "Cloud", "Sea"};
	cout << "Class Accuracy:\n";
	for (int i = 0; i < 3; i++)
	{
		float accuracy = (totalByClass[i] > 0) ? (float)correctByClass[i] / totalByClass[i] * 100 : 0;
		cout << classNames[i] << ": " << correctByClass[i] << "/" << totalByClass[i]
				 << " files correctly classified (" << fixed << setprecision(2) << accuracy << "%)\n";
	}

	// Print overall accuracy
	int totalCorrect = correctByClass[0] + correctByClass[1] + correctByClass[2];
	int totalFiles = totalByClass[0] + totalByClass[1] + totalByClass[2];
	float overallAccuracy = (totalFiles > 0) ? (float)totalCorrect / totalFiles * 100 : 0;

	cout << "\nOverall accuracy: " << totalCorrect << "/" << totalFiles
			 << " (" << fixed << setprecision(2) << overallAccuracy << "%)\n";
}

void testPatches(const string &testFolder, int trueLabel, Ptr<KNearest> &knn)
{
	cout << "==========================\n";

	cout << "Testing images in " << testFolder << "..." << endl;
	int count = 0;

	for (const auto &entry : fs::directory_iterator(testFolder))
	{
		Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);
		if (img.empty())
		{
			cout << "Warning: Could not read " << entry.path().string() << endl;
			continue;
		}

		// Make sure test images are processed the same way as training images
		resize(img, img, Size(32, 32)); // Same size as test data

		// Show image dimensions for debugging
		cout << "Testing: " << entry.path().filename() << " size: " << img.size() << endl;

		Mat hist = computeLBPHistogram(img);
		Mat results;

		// For debugging purposes, show the classification result for each file
		float response = knn->predict(hist);
		string predictedClass = (response == 0) ? "Grass" : (response == 1) ? "Cloud"
																																				: "Sea";
		cout << "File: " << entry.path().filename() << " - Predicted: " << predictedClass << endl;

		// Now do the weighted KNN classification
		Mat neighborResponses, neighborDistances;
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
		count++;
	}
	cout << "Processed " << count << " files from " << testFolder << endl;
	cout << "==========================\n";
}

void loadTrainingData(const string &folder, int label, Mat &features, Mat &labels)
{
	int count = 0;
	cout << "Loading data from: " << folder << endl;

	for (const auto &entry : fs::directory_iterator(folder))
	{
		Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);
		if (img.empty())
		{
			cout << "Warning: Could not read " << entry.path().string() << endl;
			continue;
		}

		// Use 32x32 pixels to match test data
		resize(img, img, Size(32, 32));

		// Debug: Show some image info
		cout << "Loaded: " << entry.path().filename() << " size: " << img.size() << endl;

		Mat hist = computeLBPHistogram(img);
		features.push_back(hist);
		labels.push_back(label);
		count++;
	}

	cout << "Loaded " << count << " images for class " << label << endl;
}

int main()
{
	// First, train the KNN model with training data
	Mat features, labels;

	cout << "Loading training data..." << endl;

	// Load training data
	loadTrainingData("../data/grass", grassLabel, features, labels);
	loadTrainingData("../data/cloud", cloudLabel, features, labels);
	loadTrainingData("../data/sea", seaLabel, features, labels);

	if (features.empty() || labels.empty())
	{
		cerr << "Error: No training data loaded" << endl;
		return -1;
	}

	cout << "Training KNN classifier with " << features.rows << " samples..." << endl;

	// Create and train the KNN model
	Ptr<KNearest> knn = KNearest::create();
	knn->setDefaultK(K);
	knn->train(features, ROW_SAMPLE, labels);

	cout << "KNN training complete. Running tests..." << endl;

	// Test on each directory
	testPatches("../test/grass", grassLabel, knn);
	testPatches("../test/cloud", cloudLabel, knn);
	testPatches("../test/sea", seaLabel, knn);

	printConfusionMatrix();

	// Display number of training samples per class
	int grassCount = 0, cloudCount = 0, seaCount = 0;
	for (int i = 0; i < labels.rows; i++)
	{
		if (labels.at<float>(i) == grassLabel)
			grassCount++;
		else if (labels.at<float>(i) == cloudLabel)
			cloudCount++;
		else if (labels.at<float>(i) == seaLabel)
			seaCount++;
	}

	cout << "Training data distribution:" << endl;
	cout << "- Grass: " << grassCount << " samples" << endl;
	cout << "- Cloud: " << cloudCount << " samples" << endl;
	cout << "- Sea: " << seaCount << " samples" << endl;

	// Try different K values if needed
	// knn->setDefaultK(K);

	// Display feature vector information for debugging
	cout << "Feature matrix size: " << features.rows << " samples x " << features.cols << " features" << endl;

	// Add more training information
	if (features.rows < 10)
	{
		cout << "WARNING: Very small training set. Results may be unreliable." << endl;
	}

	// Debug: Show some sample data
	if (!features.empty())
	{
		cout << "First feature vector sum: " << sum(features.row(0))[0] << endl;
	}

	return 0;
}