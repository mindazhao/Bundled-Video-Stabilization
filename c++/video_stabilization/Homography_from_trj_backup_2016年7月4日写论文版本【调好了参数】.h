#include <opencv2/opencv.hpp>
#include "opencv2/objdetect/objdetect.hpp" 
#include "opencv2/features2d/features2d.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/calib3d/calib3d.hpp" 
#include "opencv2/imgproc/imgproc_c.h" 
#include "opencv2/imgproc/imgproc.hpp"   
#include "opencv2/nonfree/features2d.hpp"
#include<opencv2/legacy/legacy.hpp>
#include <string>
#include <vector>
#include <deque>
#include <set>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <ctime> 

using namespace cv;
using namespace std;


void quick_sort(float *s, vector<Mat> &H_list, int l, int r);
float Residual(Mat H, Mat X1, Mat X2);
void Nelder_Mead(Mat &H0, Mat &pt_bg_inlier, Mat &cor_smooth_inlier, int max_iter, float eps, Mat &H, bool show_ransac);
void RANSAC_Foreground_Judgement(vector<Point2f> &pt_bg_cur, vector<Point2f> &Trj_cor_smooth, int max_iter, bool show_ransac, float scale, unsigned char *Foreground_times, int width, int height);
Mat Homography_Nelder_Mead_with_outliers(vector<Point2f> &pt_bg_cur, vector<Point2f> &Trj_cor_smooth, int max_iter, Mat& outliers, int height);
inline float cross_product(Point2f &A, Point2f &B, Point2f &C);
bool check_coefficients(Mat &H);
float Residual(Mat H, Mat X1, Mat X2);