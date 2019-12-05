/*
2018.11.16 修改了const int max_dist = 20000;//这个数据库比较特殊，差的很多，所以要设的比较大，之前特别不准，导致feature_loss 这一项很大，达到了0.15
*/



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
#include <sstream>
#include <vector>
#include <deque>
#include <set>
#include <algorithm>
#include <iostream>
//#include "CvxText.h"
#include <fstream>
#include <process.h>  
#include <windows.h>  
#include <regex>
//#include "Affine2D.hpp"
//#include  "normalization.h"
//#include "delaunay.h"
//#include "Homography_from_trj.h"
#include"Homography_from_trj_backup_2016年7月4日写论文版本【调好了参数】.h"
#include<io.h>
#include<direct.h>
using namespace cv;
using namespace std;

#define MATCH_PAUSE 1
#define H_cnstrt_error_file 0
#define OUTFILE 0
#define SHOW_BG_POINT 0
#define SHOW_DELAUNAY 0
#define OUTPUT_TRJ_COR_MAT 0
#define SHOW_DELAUNAY_BUNDLED 0
#define USE_SURF 0

//关键段（临界锁）变量声明  
//CRITICAL_SECTION  g_csThreadParameter, g_csThreadCode;  
const int reserve_times = 120;	//运行到500帧的时候，开始剔除最前面的一些轨迹
vector<Point2f> pt_init_1, pt_init_2;
vector<KeyPoint>key_init;
Mat homo_init = Mat::zeros(3, 3, CV_32F);
// 轨迹结构体，保存轨迹的属性和轨迹坐标
typedef class trajectory
{
public:
	unsigned int count;						//轨迹出现的次数
	int last_number;				//上一次出现的帧号
	unsigned int continuity;					//连续出现的帧数
	unsigned int foreground_times;		//被判为前景轨迹的次数
	unsigned int background_times;		//被判为背景轨迹次数
	unsigned int award;					//背景点奖励
	unsigned int penalization;				//前景点惩罚
	deque<Point2f> trj_cor;			//轨迹坐标
	//	Mat descriptors1;							//特征描述符，应放在一大块内存中
public:
	trajectory(unsigned int ct = 1, unsigned int ln = 1, unsigned int cnty = 1, unsigned int ft = 0, unsigned int award_time = 0, unsigned int penalization_time = 0, int n_time = reserve_times) :trj_cor(n_time)
	{
		count = ct;
		last_number = ln;
		continuity = cnty;
		foreground_times = ft;
		award = award_time;
		penalization = penalization_time;
	};
}Trj;

//设计用于计算单应矩阵的并行化
typedef class para_for_homo
{
public:
	vector<Point2f> pt_bg_cur;
	vector<Point2f> pt_smooth;
	bool start;
	unsigned char*foreground_times;
	int height;
	int width;
	para_for_homo(vector<Point2f>& pt_cur = pt_init_1, vector<Point2f>& pt_s = pt_init_2, bool actived = false, unsigned char*fg = NULL, int framewidth = 1280, int frameheight = 720)
	{
		pt_bg_cur = pt_cur;
		pt_smooth = pt_s;
		start = actived;
		foreground_times = fg;
		height = frameheight;
		width = framewidth;
	}
}PARA_FOR_HOMO;
//设计用于计算帧内距离的并行化
typedef class para_for_inframedist
{
public:
	vector<KeyPoint> cur_key;
	Mat Dist;
	bool up_left;
	para_for_inframedist(vector<KeyPoint> &keys_cur, Mat &h, bool actived)
	{
		cur_key = keys_cur;
		Dist = h;
		up_left = actived;
	}
}PARA_FOR_INFRAMEDIST;
//设计用于提取Harris角点的并行化
typedef class para_for_harris
{
public:
	vector<KeyPoint> cur_key;
	Mat img;
	GoodFeaturesToTrackDetector cur_detector;
	int up;			//第几行
	int colum;	//第几列
	float delta_y;	//纵向偏差
	float delta_x;	//横向偏差
	para_for_harris(vector<KeyPoint> &keys_cur, Mat I, GoodFeaturesToTrackDetector detector, float d_x, float d_y)
	{
		cur_key = keys_cur;
		img = I;
		cur_detector = detector;
		delta_x = d_x;
		delta_y = d_y;
	}
	para_for_harris()
	{

	}
}PARA_FOR_HARRIS;

//调用单应矩阵计算函数的多线程函数，要创建多个线程
unsigned int __stdcall Calculate_Homography(void *para_pt)
{
	//LeaveCriticalSection(&g_csThreadParameter);//离开子线程序号关键区域 
	para_for_homo *these_pt = (para_for_homo*)para_pt;
	//由于创建线程是要一定的开销的，所以新线程并不能第一时间执行到这来  
	//while(1)
	{
		while (!these_pt->start);
		if (these_pt->start)
		{
			Mat homo = Mat::zeros(3, 3, CV_32F);
			RANSAC_Foreground_Judgement(these_pt->pt_bg_cur, these_pt->pt_smooth, 100, false, 1.0, these_pt->foreground_times, these_pt->width, these_pt->height);
			these_pt->start = false;
		}
	}
	return 0;
}
//调用单应矩阵计算函数的多线程函数，要创建多个线程
unsigned int __stdcall Calculate_InFrameDistance(void *para_key)
{
	PARA_FOR_INFRAMEDIST *these_pt = (PARA_FOR_INFRAMEDIST*)para_key;
	//由于创建线程是要一定的开销的，所以新线程并不能第一时间执行到这来  
	int n_node = these_pt->cur_key.size();
	int n_node_1_2 = n_node / 2;
	vector<KeyPoint>cur_key = these_pt->cur_key;
	Mat Dij_1 = these_pt->Dist;//两个线程中会分别赋值给不同的局部变量，但是是指向同一个全局变量Dij_1，会不会造成问题？？隐患？？？
	float temp = 0.f;
	if (these_pt->up_left)
	{
		for (int i = 0; i<n_node_1_2; i++)
		{
			for (int j = i + 1; j<n_node - i; j++)	//改进，因为下面一次性赋两个值，所以循环次数也可以减半，之前这里仍然是从j=0开始的
			{
				//temp = sqrt((cur_key[i].pt.x-cur_key[j].pt.x)*(cur_key[i].pt.x-cur_key[j].pt.x) + (cur_key[i].pt.y-cur_key[j].pt.y)*(cur_key[i].pt.y-cur_key[j].pt.y));
				temp = fabs(cur_key[i].pt.x - cur_key[j].pt.x) + fabs(cur_key[i].pt.y - cur_key[j].pt.y);
				((float*)Dij_1.data)[i*n_node + j] = temp;
				((float*)Dij_1.data)[j*n_node + i] = temp;
			}
		}
	}
	else
	{
		for (int j = n_node_1_2; j<n_node; j++)
		{
			for (int i = n_node - j; i <= j; i++)	//改进，因为下面一次性赋两个值，所以循环次数也可以减半，之前这里仍然是从j=0开始的
			{
				//temp = sqrt((cur_key[i].pt.x-cur_key[j].pt.x)*(cur_key[i].pt.x-cur_key[j].pt.x) + (cur_key[i].pt.y-cur_key[j].pt.y)*(cur_key[i].pt.y-cur_key[j].pt.y));
				temp = fabs(cur_key[i].pt.x - cur_key[j].pt.x) + fabs(cur_key[i].pt.y - cur_key[j].pt.y);
				((float*)Dij_1.data)[i*n_node + j] = temp;
				((float*)Dij_1.data)[j*n_node + i] = temp;
			}
		}
	}

	return 0;
}
//调用单应矩阵计算函数的多线程函数，要创建多个线程
unsigned int __stdcall Detect_Harris(void *para_pt)
{
	//LeaveCriticalSection(&g_csThreadParameter);//离开子线程序号关键区域 
	para_for_harris *cur_detect = (para_for_harris*)para_pt;
	cur_detect->cur_detector.detect(cur_detect->img, cur_detect->cur_key);
	//进行坐标复原
	int size = cur_detect->cur_key.size();
	float d_x = cur_detect->delta_x;
	float d_y = cur_detect->delta_y;
	if (d_x>0 && d_y>0)
	{
		for (int i = 0; i<size; i++)
		{
			cur_detect->cur_key[i].pt.x += d_x;
			cur_detect->cur_key[i].pt.y += d_y;
		}
	}
	else if (d_x>0)
	{
		for (int i = 0; i<size; i++)
			cur_detect->cur_key[i].pt.x += d_x;
	}
	else if (d_y>0)
	{
		for (int i = 0; i<size; i++)
			cur_detect->cur_key[i].pt.y += d_y;
	}
	return 0;
}
//比较算法
bool compare1(const DMatch &d1, const  DMatch &d2)
{
	return d1.trainIdx < d2.trainIdx;
}
//计算平均值
float mean(const deque<int> trj_num)
{
	float sum = 0;
	int trjs = trj_num.size();
	for (int i = 0; i<trjs; i++)
		sum += trj_num[i];
	sum /= trjs;
	return sum;
}
//计算标准差，除数为(N-1)
float std_val(const deque<int> trj_num, float the_mean)
{
	float std_var = 0;
	int trjs = trj_num.size();
	for (int i = 0; i<trjs; i++)
		std_var += (trj_num[i] - the_mean) * (trj_num[i] - the_mean);
	std_var /= (trjs - 1);
	return sqrt(std_var);
}
// 计算汉明距离
unsigned int hamdist2(unsigned char* a, unsigned char* b, size_t size)
{
	HammingLUT lut;
	unsigned int result;
	result = lut((a), (b), size);
	return result;
}
// 该函数只用到Trj_keys.size()和cur_key.size()函数，只用到二者的长度!
// 为了更高效，应单独设计一个vector存放所有的描述符
void naive_nn_search(Mat& descp1, Mat& descp2, vector<DMatch>& matches)
{
	//vector<unsigned int> matched_cur, matched_Trj;	//保存匹配上的点对序号
	int cur_key_size = descp2.rows;
	int Trj_keys_size = descp1.rows;
	for (int i = 0; i < cur_key_size; i++)
	{
		unsigned int min_dist = INT_MAX;
		unsigned int sec_dist = INT_MAX;
		int min_idx = -1, sec_idx = -1;
		unsigned char* query_feat = descp2.ptr(i);
		for (int j = 0; j < Trj_keys_size; j++)
		{
			unsigned char* train_feat = descp1.ptr(j);
			unsigned int dist = hamdist2(query_feat, train_feat, 64); //修正了一个严重的错误！这个是64位的FREAK算子！
			//最短距离
			if (dist < min_dist)
			{
				sec_dist = min_dist;
				sec_idx = min_idx;
				min_dist = dist;
				min_idx = j;
			}
			//次短距离
			else if (dist < sec_dist)
			{
				sec_dist = dist;
				sec_idx = j;
			}
		}
		if (min_dist <= 50 && min_dist <= 0.8*sec_dist)//min_dist <= (unsigned int)(sec_dist * 0.7) && min_dist <=100
		{
			//若不处理，则会有许多重复的匹配对！
			bool repeat = false;
			if (matches.size()>0)
			{
				for (int k = 0; k<matches.size(); k++)
				{
					if (min_idx == matches.at(k).trainIdx)
						repeat = true;
				}
				if (!repeat)
					matches.push_back(DMatch(i, min_idx, 0, (float)min_dist));
			}
			else matches.push_back(DMatch(i, min_idx, 0, (float)min_dist));
		}
	}
}
// 该函数只用到Trj_keys.size()和cur_key.size()函数，只用到二者的长度!
// 为了更高效，应单独设计一个vector存放所有的描述符
void naive_nn_search2(vector<KeyPoint>& Trj_keys, Mat& descp1, vector<KeyPoint>& cur_key, Mat& descp2, vector<vector<DMatch>>& matches, const int max_shaky_dist, int k)
{
	//vector<unsigned int> matched_cur, matched_Trj;	//保存匹配上的点对序号
	int cur_key_size = cur_key.size();
	int Trj_keys_size = Trj_keys.size();
	//cout << max_shaky_dist << endl;
	for (int i = 0; i < cur_key_size; i++)
	{
		unsigned int min_dist = INT_MAX;
		unsigned int sec_dist = INT_MAX;
		unsigned int thr_dist = INT_MAX;
		int min_idx = -1, sec_idx = -1, thr_idx = -1;
		unsigned char* query_feat = descp2.ptr(i);
		float cur_key_x = cur_key[i].pt.x;
		float cur_key_y = cur_key[i].pt.y;
		for (int j = 0; j < Trj_keys_size; j++)
		{
			unsigned char* train_feat = descp1.ptr(j);
			unsigned int dist = hamdist2(query_feat, train_feat, 128); //修正了一个严重的错误！这个是64位的FREAK算子！
			float Trj_key_x = Trj_keys[j].pt.x;
			float Trj_key_y = Trj_keys[j].pt.y;
			//匹配点坐标距离限制
			if ((cur_key_x - Trj_key_x)*(cur_key_x - Trj_key_x) + (cur_key_y - Trj_key_y)*(cur_key_y - Trj_key_y) < max_shaky_dist)
			{
				//cout << cur_key_x << "  " << Trj_key_x << endl;
				//最短距离
				if (dist < min_dist)
				{
					thr_dist = sec_dist;
					thr_idx = sec_idx;
					sec_dist = min_dist;
					sec_idx = min_idx;
					min_dist = dist;
					min_idx = j;
				}
				//次短距离
				else if (dist < sec_dist)
				{
					thr_dist = sec_dist;
					thr_idx = sec_idx;
					sec_dist = dist; sec_idx = j;
				}
				//次短距离
				else if (dist < thr_dist)
				{
					thr_dist = dist; thr_idx = j;
				}
			}
		}
		if (min_dist <= 125)
		{
			matches[i].push_back(DMatch(i, min_idx, 0, (float)min_dist));
			if (k>1)
				matches[i].push_back(DMatch(i, sec_idx, 0, (float)sec_dist));
			if (k>2)
				matches[i].push_back(DMatch(i, thr_idx, 0, (float)thr_dist));
		}
		else
			matches[i].push_back(DMatch(i, -1, 0, (float)min_dist));
	}
}
//快排
void quick_sort(Mat v, vector<Point> &L, int l, int r)
{
	if (l < r)
	{
		int i = l, j = r;
		float x = ((float*)v.data)[l];
		Point temp = L[l];
		while (i < j)
		{
			while (i < j && ((float*)v.data)[j] <= x) // 从右向左找第一个小于x的数  
				j--;
			if (i < j)
			{
				((float*)v.data)[i++] = ((float*)v.data)[j];
				L[i - 1] = L[j];
			}

			while (i < j && ((float*)v.data)[i] > x) // 从左向右找第一个大于等于x的数  
				i++;
			if (i < j)
			{
				((float*)v.data)[j--] = ((float*)v.data)[i];
				L[j + 1] = L[i];
			}
		}
		((float*)v.data)[i] = x;
		L[i] = temp;
		quick_sort(v, L, l, i - 1); // 递归调用   
		quick_sort(v, L, i + 1, r);
	}
}


//my_spectral_matching，根据自己编写的matlab函数编写
void my_spectral_matching(vector<KeyPoint> &cur_key, vector<KeyPoint> &last_key, vector<vector<DMatch>> &matches, int &k, vector<DMatch> &X_best)
{
	int64 st, et;
	//st = cvGetTickCount();
	int n_node = cur_key.size();
	int n_label = last_key.size();
	vector<int> start_ind_for_node(n_node);	//每个拥有有效匹配对的当前特征点在L中的起始索引值
	int n_matches = 0;	//有效匹配对个数
	vector<Point> L;	//所有的候选匹配对
	for (int i = 0; i<n_node; i++)
	{
		if (matches[i][0].trainIdx != -1)
			start_ind_for_node[i] = n_matches;
		else
		{
			start_ind_for_node[i] = -1;
			continue;
		}
		for (int j = 0; j<k; j++)
		{
			if (matches[i][j].trainIdx != -1)
			{
				L.push_back(Point(i, matches[i][j].trainIdx));
				n_matches++;
			}
			//else
			//	cout<<i<<"\t"<<j<<endl;
		}
	}
	//et = cvGetTickCount();
	//printf("生成L矩阵时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 
	////%% 计算M的对角线元素
	//st = cvGetTickCount();
	Mat M = Mat::zeros(n_matches, n_matches, CV_32F);
	int n_cur = 0;
	float Ham_max = 256.0;
	int tmp = 0;
	for (int i = 0; i<n_node - 1; i++)
	{
		if (start_ind_for_node[i] != -1)
		{
			n_cur = start_ind_for_node[i + 1] - start_ind_for_node[i];
			for (int j = 0; j<n_cur; j++)
			{
				tmp = start_ind_for_node[i] + j;
				((float*)M.data)[tmp*n_matches + tmp] = matches[i][j].distance / Ham_max;
			}
		}
	}
	if (start_ind_for_node[n_node - 1] != -1)
	{
		n_cur = n_matches - start_ind_for_node[n_node - 1];
		for (int j = 0; j<n_cur; j++)
		{
			tmp = start_ind_for_node[n_node - 1] + j;
			((float*)M.data)[tmp*n_matches + tmp] = matches[n_node - 1][j].distance / Ham_max;
		}
	}
	//et = cvGetTickCount();
	//printf("生成M矩阵对角线元素时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 

	////先计算每帧内每个i,j之间的坐标距离
	//st = cvGetTickCount();
	Mat Dij_1 = Mat::zeros(n_node, n_node, CV_32F);
	Mat Dij_2 = Mat::zeros(n_label, n_label, CV_32F);
	float temp = 0;

	//多线程版本
	//st = cvGetTickCount();
	const int THREAD_NUM = 4;
	HANDLE handle[THREAD_NUM];
	PARA_FOR_INFRAMEDIST pthread_array_1 = PARA_FOR_INFRAMEDIST(cur_key, Dij_1, true);
	PARA_FOR_INFRAMEDIST pthread_array_2 = PARA_FOR_INFRAMEDIST(cur_key, Dij_1, false);
	PARA_FOR_INFRAMEDIST pthread_array_3 = PARA_FOR_INFRAMEDIST(last_key, Dij_2, true);
	PARA_FOR_INFRAMEDIST pthread_array_4 = PARA_FOR_INFRAMEDIST(last_key, Dij_2, false);
	handle[0] = (HANDLE)_beginthreadex(NULL, 0, Calculate_InFrameDistance, &pthread_array_1, 0, NULL);
	handle[1] = (HANDLE)_beginthreadex(NULL, 0, Calculate_InFrameDistance, &pthread_array_2, 0, NULL);
	handle[2] = (HANDLE)_beginthreadex(NULL, 0, Calculate_InFrameDistance, &pthread_array_3, 0, NULL);
	handle[3] = (HANDLE)_beginthreadex(NULL, 0, Calculate_InFrameDistance, &pthread_array_4, 0, NULL);

	WaitForMultipleObjects(THREAD_NUM, handle, TRUE, INFINITE);//INFINITE);//至多等待20ms
	for (int i = 0; i<4; i++)
		CloseHandle(handle[i]);
	//et = cvGetTickCount();
	//printf("计算帧内距离时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 

	//计算M矩阵的非对角线元素
	//st = cvGetTickCount();
	float sigma = 4;
	float sigma_d_3 = sigma * 3;
	float delta_2 = 2 * sigma*sigma;
	float tmp1 = 0, tmp2 = 0, tmp3 = 0;
	for (int i = 1; i<n_matches; i++)
	{
		for (int j = i + 1; j<n_matches; j++)
		{
			tmp1 = ((float*)Dij_1.data)[(L[i].x)*n_node + L[j].x];
			tmp2 = ((float*)Dij_2.data)[(L[i].y)*n_label + L[j].y];
			temp = tmp1 - tmp2;
			if (fabs(temp) < sigma_d_3)
			{
				tmp3 = 4.5 - (temp*temp) / delta_2;
				((float*)M.data)[i*n_matches + j] = tmp3;
				((float*)M.data)[j*n_matches + i] = tmp3;
			}
		}
	}
	//et = cvGetTickCount();
	//printf("生成M矩阵非对角线元素时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 

	//%% spectral matching算法
	//st = cvGetTickCount();
	Mat v = Mat::ones(n_matches, 1, CV_32F);
	//float x = norm(v);
	v = v / norm(v);
	int iterClimb = 20;//之前取30，耗时太长，取20应该也可以，反正后面也会收敛了

	// 幂法计算最大特征值（及其）对应的特征向量
	for (int i = 0; i<iterClimb; i++)
	{
		v = M*v;
		v = v / norm(v);
	}
	//et = cvGetTickCount();
	//printf("幂法计算主特征向量时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 
	//贪心策略求出最优匹配
	//st = cvGetTickCount();
	//vector<DMatch> X_best;
	quick_sort(v, L, 0, n_matches - 1);
	//et = cvGetTickCount();
	//printf("快速排序时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 
	float max_v = 0;
	//DMatch best_match;
	//st = cvGetTickCount();
	bool *conflict = new bool[n_matches];	//标记每个匹配对是否与当前最好的匹配对冲突
	for (int i = 0; i<n_matches; i++)
		conflict[i] = false;
	int left_matches = n_matches;
	float dist = 10.0;
	while (left_matches)
	{
		int i = 0;
		while (conflict[i]) i++;	//找到第一个未冲突的最大值项
		max_v = ((float*)v.data)[i];
		DMatch best_match = DMatch(L[i].x, L[i].y, 0, float(dist));
		X_best.push_back(best_match);
		//找出所有与best_match冲突的匹配对，剔除之
		for (int j = 0; j<n_matches; j++)
		{
			if ((L[j].x == best_match.queryIdx || L[j].y == best_match.trainIdx) && !conflict[j])
			{
				conflict[j] = true;
				left_matches--;
			}
		}
	}
	delete[]conflict;
	//et = cvGetTickCount();
	//printf("贪心策略时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.); 
}
/*
pts，要剖分的散点集,in
img,剖分的画布,in
tri,存储三个表示顶点变换的正数,out
*/
// used for doing delaunay trianglation with opencv function
//该函数用来防止多次重画并消去虚拟三角形的顶点
bool isGoodTri(Vec3i &v, vector<Vec3i> & tri)
{
	int a = v[0], b = v[1], c = v[2];
	v[0] = min(a, min(b, c));//v[0]找到点插入的先后顺序（0....N-1，N为点的个数）的最小值
	v[2] = max(a, max(b, c));//v[2]存储最大值.
	v[1] = a + b + c - v[0] - v[2];//v[1]为中间值
	if (v[0] == -1) return false;

	vector<Vec3i>::iterator iter = tri.begin();//开始时为空
	for (; iter != tri.end(); iter++)
	{
		Vec3i &check = *iter;//如果当前待压入的和存储的重复了，则停止返回false。
		if (check[0] == v[0] &&
			check[1] == v[1] &&
			check[2] == v[2])
		{
			break;
		}
	}
	if (iter == tri.end())
	{
		tri.push_back(v);
		return true;
	}
	return false;
}
//二分查找，因为foreground_index是从小到大排好序的
int binary_search(vector<int>a, int goal)
{
	if (a.size() == 0)
	{
		cout << "传入vector是空的，查个屁啊" << endl;
		return -1;
	}
	int low = 0;
	int high = a.size() - 1;
	while (low <= high)
	{
		int middle = (low + high) / 2;
		if (a[middle] == goal)
			return middle;
		//在左半边
		else if (a[middle] > goal)
			high = middle - 1;
		//在右半边
		else
			low = middle + 1;
	}
	//没找到
	return -1;
}
vector<float> MyGauss(int _sigma)
{
	int width = 2 * _sigma + 1;
	if (width<1)
	{
		width = 1;
	}


	/// 设定高斯滤波器宽度
	int len = width;

	/// 高斯函数G
	vector<float> GassMat;

	int cent = len / 2;
	float summ = 0;
	for (int i = 0; i<len; i++)
	{
		int radius = (cent - i) * (cent - i);
		GassMat.push_back(exp(-((float)radius) / (2 * _sigma * _sigma)));

		summ += GassMat[i];
	}
	for (int i = 0; i<len; i++)
		GassMat[i] /= (summ + 0.001);

	return GassMat;
}
//FAST阈值比较函数
bool fast_thresh_comp(const KeyPoint &corner1, const KeyPoint &corner2)
{
	return corner1.response > corner2.response;
}
//限制两个特征点距离的FAST特征检测算法
void MyFAST(Mat& image, vector<KeyPoint>& corners, int maxCorners, double qualityLevel, double minDistance, Mat & mask, bool have_mask)
{
	vector<KeyPoint>FAST_corners;
	//基本FAST检测
	FAST(image, corners, qualityLevel);

	//在Mask基础上生成一张掩码表
	Mat temp_mask(image.size(), CV_8U, Scalar(0));
	int num_corners = corners.size();
	int width = image.cols;
	int height = image.rows;
	if (have_mask)
	{
		for (int i = 0; i<num_corners; i++)
		{
			if (((unsigned char*)mask.data)[cvRound(corners[i].pt.x + corners[i].pt.y*width)])
			{
				((unsigned char*)temp_mask.data)[cvRound(corners[i].pt.x + corners[i].pt.y*width)] = 255;
				FAST_corners.push_back(corners[i]);
			}
		}
	}
	else
	{
		FAST_corners = corners;
		for (int i = 0; i<num_corners; i++)
		{
			((unsigned char*)temp_mask.data)[cvRound(corners[i].pt.x + corners[i].pt.y*width)] = 255;
		}
	}

	num_corners = FAST_corners.size();
	corners.clear();
	//贪心算法，筛选
	sort(FAST_corners, fast_thresh_comp);
	for (int i = 0; i<num_corners && corners.size()<maxCorners; i++)
	{
		if (((unsigned char*)temp_mask.data)[cvRound(FAST_corners[i].pt.x + FAST_corners[i].pt.y*width)])
		{
			int rw_right = min(cvRound(FAST_corners[i].pt.x) + minDistance, width);
			int rw_left = max(cvRound(FAST_corners[i].pt.x) - minDistance, 0);
			int cl_up = max(cvRound(FAST_corners[i].pt.y) - minDistance, 0);
			int cl_down = min(cvRound(FAST_corners[i].pt.y) + minDistance, height);

			//cout<<temp_mask.rowRange(rw_left, rw_right).colRange(cl_up, cl_down)<<endl;
			temp_mask.colRange(rw_left, rw_right).rowRange(cl_up, cl_down) = Mat::zeros(rw_right - rw_left, cl_down - cl_up, CV_8U);
			corners.push_back(FAST_corners[i]);
		}
	}
	//cout<<corners.size()<<endl;
}
//argv格式：程序名 视频文件名 竖直方向检测特征点起点线y坐标 竖直方向检测特征点终点线y坐标 帧率


void getAllFiles(string path, vector<string>& files, string format)
{
	long long hFile = 0;//文件句柄  64位下long 改为 intptr_t
	struct _finddata_t fileinfo;//文件信息 
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*" + format).c_str(), &fileinfo)) != -1) //文件存在
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR))//判断是否为文件夹
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)//文件夹名中不含"."和".."
				{
					files.push_back(p.assign(path).append("\\").append(fileinfo.name)); //保存文件夹名
					getAllFiles(p.assign(path).append("\\").append(fileinfo.name), files, format); //递归遍历文件夹
				}
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));//如果不是文件夹，储存文件名
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}



int main(int argc, char* argv[])
{


	char* class_name[3] = { "Regular", "Crowd", "Parallax" };
	int OutputPadding = 200;
	for (int cl = 0; cl < 3; cl++)
	{

		char filePath_unstable[40] = "E:\\data\\BUNDLED2\\results_images\\";
		strcat(filePath_unstable, class_name[cl]);

		string num_trj = "100";

		char path_videos[40];
		strcpy(path_videos, filePath_unstable);
		//strcat(path_videos, "\\videos");

		//strcpy(filePath_unstable, "E:\\data\\stable\\");
		//strcat(filePath_unstable, cl.c_str());
		//char * filePath_unstable = strcat(, );
		////获取该路径下的所有文件  
		vector<string>files_unstable;
		string format = ".avi";				 //查找文件的格式
		//getAllFiles(filePath_stable, files_stable, format);
		getAllFiles(path_videos, files_unstable, format);

		//int numframe[] = { 256, 513, 492, 844,478, 248, 527, 528, 572, 597, 432 };
		for (int i = 0; i < files_unstable.size(); i++)
		{
			vector<string>bmp;
			getAllFiles(files_unstable[i], bmp, ".bmp");



			int the_start_num = files_unstable[i].find_last_of("\\");
			string path_temp = files_unstable[i].substr(the_start_num + 1);//, filename.length()-4);
			//提取帧率和大小

			string path_unstable_video = "E:\\data\\PROPOSED\\unstable\\" + string(class_name[cl]) + "\\videos\\" + path_temp;
			VideoCapture capture;
			capture.open(path_unstable_video);
			int fps = capture.get(CV_CAP_PROP_FPS);
			int width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
			int height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);

			string path_result = "E:\\data\\BUNDLED2\\results\\" + string(class_name[cl]) + "\\" + path_temp;

			VideoWriter writer_dense;
			writer_dense.open(path_result, CV_FOURCC('X', 'V', 'I', 'D'), fps, Size(width, height), true);
			for (int j = 0; j < bmp.size(); j++)
			{
				string path = files_unstable[i] + "\\"+ to_string(j + 1) + ".bmp";
				Mat frame = imread(path);

				Mat frame_crop = frame.rowRange(OutputPadding, frame.size().height - OutputPadding).colRange(OutputPadding, frame.size().width - OutputPadding);
				writer_dense << frame;
				imshow("frame_crop", frame_crop);
				imshow("frame_origin", frame);
				waitKey(1);
				cout << class_name[cl] << "   " << path_temp << "   " << j <<"/"<< bmp.size() << endl;
			}

			capture.release();


		}
	}


	/*string path_videos = "E:\\Bundled\\results\\";

	VideoWriter writer_dense;
	writer_dense.open("bd.avi", CV_FOURCC('X', 'V', 'I', 'D'), 30, Size(640, 360), true);
	for (int i = 0; i < 246; i++)
	{
	string path = path_videos + to_string(i + 1) + ".bmp";
	Mat frame = imread(path);
	writer_dense << frame;
	cout << i << endl;
	}
	writer_dense.release();*/
	return 0;
}