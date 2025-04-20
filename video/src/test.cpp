#include <opencv2/opencv.hpp>
#include <iostream>
#include <map>

using namespace cv;
using namespace std;

struct Vec3bComparator
{
	bool operator()(const Vec3b &a, const Vec3b &b) const
	{
		if (a[0] != b[0])
			return a[0] < b[0];
		if (a[1] != b[1])
			return a[1] < b[1];
		return a[2] < b[2];
	}
};

map<Vec3b, string, Vec3bComparator> eightColorMap = {
		{{0, 0, 0}, "black"},				// black
		{{255, 0, 0}, "blue"},			// blue (BGR)
		{{0, 255, 0}, "green"},			// green
		{{255, 255, 0}, "cyan"},		// cyan
		{{0, 0, 255}, "red"},				// red
		{{255, 0, 255}, "magenta"}, // magenta
		{{0, 255, 255}, "yellow"},	// yellow
		{{255, 255, 255}, "white"}	// white
};

Vec3b findClosestColor(Vec3b pixel)
{
	Vec3b closestColor = {0, 0, 0};
	int minDist = INT_MAX;

	for (const auto &entry : eightColorMap)
	{
		Vec3b ref = entry.first;
		int db = pixel[0] - ref[0];
		int dg = pixel[1] - ref[1];
		int dr = pixel[2] - ref[2];

		// Weighted distance (tweak as needed)
		int dist = 2 * dr * dr + 4 * dg * dg + 1 * db * db;

		if (dist < minDist)
		{
			minDist = dist;
			closestColor = ref;
		}
	}
	return closestColor;
}

int main()
{
	Vec3b testPixel = Vec3b(254, 1, 0); // BGR format (Blue=254, Green=1, Red=0)
	Vec3b result = findClosestColor(testPixel);

	cout << "Test Pixel (BGR): (" << (int)testPixel[0] << ", " << (int)testPixel[1] << ", " << (int)testPixel[2] << ")" << endl;
	cout << "Closest Color Match: (" << (int)result[0] << ", " << (int)result[1] << ", " << (int)result[2] << ") → " << eightColorMap[result] << endl;

	return 0;
}
