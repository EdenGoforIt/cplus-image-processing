#include <iostream>
#include <opencv2/opencv.hpp>

#include <unordered_set>

using namespace cv;
using namespace std;

typedef pair<int, int> pairs;

int main(int argc, char **argv)
{
	// if no arguments are given, argc - 1 evaluates to 0, so argv[0] (program name) is accessed.
	// void simply evaluate the expression and discard the result
	(void)argv[argc - 1];

	// Seed to random 100
	srand(100);

	// Read images using 3 vector 3 bytes. rgb
	Mat_<Vec3b> binary;
	binary = imread("../Binary1.jpg");

	// Convert it to gray
	Mat_<uint8_t> i;
	cvtColor(binary, i, COLOR_BGR2GRAY);

	// Display gray image on the window
	imshow("input", i);

	vector<set<pairs>> sets;
	vector<Vec3b> colours;

	int counter = -1;
	int s1, s2;
	Mat_<int> A(i.rows, i.cols);
	A = -1;
	// Iterate the ROW. Start by 1 because there is no adjacency in the first row and first column
	for (int y = 1; y < A.rows; y++)
	{
		// Iterate the COLUMN
		cout << "Y: " << y << endl;
		for (int x = 1; x < A.cols; x++)
		{
			// If pixel is the white (255) or more like white grayish object
			if (i(y, x) > 128)
			{
				// Check left and the top neighbors
				// Y is Row, X is Column
				if (i(y, x - 1) > 128 || i(y - 1, x) > 128)
				{
					s1 = A(y, x - 1); // Left neighbor's label
					s2 = A(y - 1, x); // Top neighbor's label

					// If left neighbor has a label
					if (s1 != -1)
					{
						sets[s1].insert(pair(y, x));
						// Label the current pixl with the same label
						A(y, x) = s1;
					}

					// If top neighbor has a label
					if (s2 != -1)
					{
						sets[s2].insert(pair(y, x));

						// Label the current pixel with the same label
						A(y, x) = s2;
					}

					// If pixel connects two different labeled regions
					if (s1 != s2 && s1 != -1 && s2 != -1)
					{
						// Union, Merge sets
						for (set<pairs>::iterator it = sets[s2].begin(); it != sets[s2].end(); it++)
						{
							sets[s1].insert(*it);
							A(it->first, it->second) = s1;
						}

						sets[s2].clear();
					}
				}
				else
				{
					counter++;
					set<pairs> ns;
					sets.push_back(ns);
					sets[counter].insert(pair(y, x));

					// Generate random color for visualization (RGB values between 35-255)
					colours.push_back(Vec3b(rand() % 220 + 35, rand() % 220 + 35, rand() % 220 + 35));
					A(y, x) = counter;
				}
			}
		}
	}

	cout << "Counter: " << counter << endl;
	imwrite("op.png", A);
	Mat_<Vec3b> c(i.rows, i.cols);
	for (int y = 0; y < i.rows; y++)
	{
		for (int x = 0; x < i.cols; x++)
			c(y, x) = colours[A(y, x)];
	}
	imwrite("opc.png", c);
	imshow("output", c);

	waitKey(0);
}