
#include <iostream>   // for standard I/O  
#include <string>   // for strings  
#include <iomanip>  // for controlling float print precision  
#include <sstream>  // string to number conversion  

#include<opencv2/contrib/contrib.hpp>
#include<opencv2/ml/ml.hpp>
#include<opencv2/objdetect/objdetect.hpp>
#include <fstream>  

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur  
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)  
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O  
using namespace std;
using namespace cv;

Size imageSize = Size(320, 180);//原始尺寸320*180
Size testSize = Size(64, 36);
Mat outImage;
CvSVM *mySVM = new CvSVM();
string svmName = "mysvm.xml";

void RGBtoHSV(float b, float g, float r, float &h, float &s, float &v)//将RGB值转化为HSV值  
{
	float max = (((r > b) ? r : b) > g) ? ((r > b) ? r : b) : g;
	float min = (((r > b) ? b : r) > g) ? g : ((r > b) ? b : r);

	if (max == min)
	{
		h = 0; s = 0; v = max / 255;
	}
	else if (g >= b)
	{
		h = (max - r + g - min + b - min) / (max - min) * 60;
		s = 1 - min / max;
		v = max / 255;
	}
	else if (g < b)
	{
		h = 360 - (max - r + g - min + b - min) / (max - min) * 60;
		s = 1 - min / max;
		v = max / 255;
	}
}

//以HSV总量作为特征量提取关键帧  
int HSV(Mat img)
{

	//  IplImage * ipl_img = NULL;  
	//  * ipl_img =img;//将Mat类型转化为IplImage  
	float B, G, R, H = 0, S = 0, V = 0;
	int nr = img.rows; // number of rows    
	int nc = img.cols; // number of cols  
	float sumH = 0, sumS = 0, sumV = 0;
	CvScalar cs;

	//将Mat类型转化为IplImage  
	IplImage *ipl_img = cvCreateImage(cvSize(nr, nc), 8, 3);
	ipl_img->imageData = (char *)img.data;

	for (int j = 0; j<nr; j++) {
		for (int i = 0; i<nc; i++) {
			cs = cvGet2D(ipl_img, i, j);
			B = cs.val[0];
			G = cs.val[1];
			R = cs.val[2];
			RGBtoHSV(B, G, R, H, S, V);
			sumH = sumH + H;
			sumS = sumS + S;
			sumV = sumV + V;
		}
	}
	float aveH = sumH / (nc*nr);
	float aveS = sumS / (nc*nr);
	float aveV = sumV / (nc*nr);
	float totalHSV = 9 * aveH + 3 * aveS + aveV;//totalHSV为特征量  
	return totalHSV;
}

void coumputeHog(const Mat& src, vector<float> &descriptors)
{
	HOGDescriptor myHog = HOGDescriptor(testSize, Size(32, 18), cvSize(16, 9), cvSize(4, 3), 9);
	myHog.compute(src.clone(), descriptors, Size(1, 1), Size(0, 0));
}

int main()
{
	VideoCapture cap;
	cap.open("test.avi");
	Mat frame[500];
	Mat outImage[500];
	Mat keyframe[500];
	int keycount = 0;
	int frmNum = 0;
	float curhsv, lathsv;//当前帧的hsv值与后一帧的hsv值  

	if (!cap.isOpened())
	{
		cout << "无法打开视频！" << endl;
		return -1;
	}


	int th = 14;//提取关键帧的阈值  
			   //  cout << "输入阈值提取关键帧(大于1)：";  
			   //  cin >> th;  
	for (;;)
	{
		frmNum++;
		cout << frmNum << endl;
		cap >> frame[frmNum];
		if (frame[frmNum].empty())
		{
			cout << "读取视频完毕 !" << endl;
			break;
		}
		resize(frame[frmNum], outImage[frmNum], imageSize);
		keyframe[0] = outImage[1];
		if (frmNum > 1)
		{
			curhsv = HSV(outImage[frmNum - 1]);
			lathsv = HSV(outImage[frmNum]);
			if (abs(curhsv - lathsv) > th)//设定阈值th提取关键帧  
			{
				keycount++;
				keyframe[keycount] = outImage[frmNum];
			}
		}
	}

	cout << "该视频总共有" << frmNum << "帧" << endl;
	cout << "在阈值为" << th << "时提取了" << keycount + 1 << " 个关键帧!" << endl;

	int arrlabel[50];
	arrlabel[0] = 0;

	for (int i = 0; i < keycount + 1; i++)
	{
		imshow("关键帧", keyframe[i]);//展示所有提取的关键帧
		mySVM->load("mysvm.xml");

		////////////////匹配识别////////////////////////
		resize(keyframe[i], keyframe[i], testSize);
		vector<float> vecDescriptors;
		coumputeHog(keyframe[i], vecDescriptors);
		Mat tempRow = ((Mat)vecDescriptors).t();
		float label = mySVM->predict(tempRow, false);
		arrlabel[i + 1] = label;
		string lab;
		switch (int(label))
		{
		case 1:
			lab = "Ni_1";
			if (arrlabel[i] != arrlabel[i + 1])
			{
				printf("你\n");
			}
			break;
		case 2:
			lab = "Hao_2";
			if (arrlabel[i] != arrlabel[i + 1])
			{
				printf("好\n");
			}
			break;
		case 3:
			lab = "F_3";
			if (arrlabel[i] != arrlabel[i + 1])
			{
				printf("F\n");
			}
			break;
		case 4:
			lab = "P_4";
			if (arrlabel[i] != arrlabel[i + 1])
			{
				printf("P\n");
			}
			break;
		case 5:
			lab = "G_5";
			if (arrlabel[i] != arrlabel[i + 1])
			{
				printf("G\n");
			}
			break;
		case 6:
			lab = "A_6";
			if (arrlabel[i] != arrlabel[i + 1])
			{
				printf("A\n");
			}
			break;
		default:
			printf("No matching!!\n");
			break;
		}
		resize(keyframe[i], keyframe[i], imageSize);
		if (arrlabel[i] != arrlabel[i + 1])
		{
			imshow(lab, keyframe[i]);
		}
		cv::waitKey(100);
	}
	return 0;
}