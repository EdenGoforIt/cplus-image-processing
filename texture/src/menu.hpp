#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace chrono;

class Menu
{
private:
	string menutitle;
	vector<string> menuitems;
	Mat_<Vec3b> image;
	int width = 180;
	int highlighted = -1;
	int selected = -1;
	Vec3b bg = Vec3b(0xbf, 0xbe, 0xbb);
	Vec3b grey = Vec3b(0xef, 0xf0, 0xf1);
	Vec3b blue = Vec3b(0xe9, 0xae, 0x3d);
	Vec3b black = Vec3b(0, 0, 0);
	Vec3b white = Vec3b(0xff, 0xff, 0xff);

public:
	Menu(string title, vector<string> items)
	{
		menutitle = title;
		menuitems = items;
		image = Mat_<Vec3b>(menuitems.size() * 32, width);
		namedWindow(menutitle, WINDOW_GUI_NORMAL | WINDOW_AUTOSIZE);
		setMouseCallback(menutitle, onmouse, (void *)this);
		drawmenu();
	}
	static void onmouse(int event, int, int y, int, void *userdata)
	{
		Menu *m = (Menu *)userdata;
		if (event == 0)
			m->sethighlighted(y / 32);
		if (event == 1)
			m->setselected(y / 32);
		m->drawmenu();
	}
	void setselected(int i)
	{
		selected = i;
		if (i >= 0 && menuitems[i] == "Exit")
			exit(1);
	}
	string getselected()
	{
		if (selected == -1)
			return "";
		return menuitems[selected];
	}
	void sethighlighted(int i)
	{
		highlighted = i;
	}
	void drawmenu()
	{
		image = bg;
		for (int i = 0; i < (int)menuitems.size(); i++)
		{
			Vec3b bgcol = grey;
			Vec3b fgcol = black;
			if (i == highlighted)
			{
				bgcol = blue;
				fgcol = white;
			}
			if (i == selected)
			{
				bgcol = black;
				fgcol = white;
			}
			rectangle(image, Rect(1, i * 32, width - 2, 32), bgcol, FILLED);
			putText(image, menuitems[i], Vec2i(12, 20 + i * 32), FONT_HERSHEY_PLAIN, 1,
							fgcol, 1, LINE_AA);
		}
		imshow(menutitle, image);
	}
};