/*
2015.6.17，编写完成，使用RANSAC+Nelder-Mead算法搜索最佳单应矩阵，并强制其非刚性参数为0
2015.6.17，添加参数限制，倾斜参数、放缩参数不得超出一定阈值
2015.6.18，修改RANSAC算法部分，对于匹配数最多的模型，按照其总体误差值进行排序，取其优者
2015.6.31，6参数改为8参数
2015.7.1，将收敛条件加强，在计算矩阵时候，将float转为float
2015.7.2，修正：Homography_Nelder_Mead中的矩阵A在每次使用前需要清零，因为是+=而不是=，所以每次清零或者声明为局部变量，每次RASAC循环中都重新定义
2015.7.20，在RANSAC部分，更改选点策略，检测随机选择的4个点是否构成一个四边形，防止其中三点甚至四点共线
2015.9.17，添加了DLT+SVD计算单应矩阵的算法，但是误差太大。。。
2015.9.24，Homography_Nelder_Mead_with_outliers多线程计算RANSAC
2015.10.30，参考matlab版本，用random_shuffle算法代替之前的使用四个随机数产生随机抽样序列，效果分分钟好了，内点个数跟matlab版本一样多了
2015.12.15，参考Goldstein，每次RANSAC过程中限制每个块的点数
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
#include <vector>
#include <deque>
#include <set>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <ctime> 
#include <set>
#include <process.h>  
#include <windows.h>  
//#include "Homography_from_trj.h"
#include"Homography_from_trj_backup_2016年7月4日写论文版本【调好了参数】.h"
using namespace cv;
using namespace std;

//归一化算法
void normalization(Mat &X, Mat &X_norm, Mat &norm_mat)
{
	int N = X.cols;
	//形心坐标
	float x0=0, y0=0;//Point2f centroid;
	for(int i=0; i<N; i++)
	{
		x0 += ((float *)X.data)[i];
		y0 += ((float *)X.data)[i+N];
	}
	x0 /= N;
	y0 /= N;
	//计算到形心的距离
	float mean_dist = 0;
	for(int i=0; i<N; i++)
		mean_dist += sqrt((((float *)X.data)[i]-x0)*(((float *)X.data)[i]-x0) + (((float *)X.data)[i+N]-y0)*(((float *)X.data)[i+N]-y0));
	mean_dist /=N;
	float sqrt_2 = sqrt(2.f);
	float mat_0_0 = sqrt_2/mean_dist;
	float mat_0_2 = -1*sqrt_2/mean_dist*x0;
	float mat_1_2 = -1*sqrt_2/mean_dist*y0;
	norm_mat.row(0).col(0) = mat_0_0;
	norm_mat.row(0).col(1) = 0;
	norm_mat.row(0).col(2) = mat_0_2;
	norm_mat.row(1).col(0) = 0;
	norm_mat.row(1).col(1) = mat_0_0;
	norm_mat.row(1).col(2) = mat_1_2;
	norm_mat.row(2).col(0) = 0;
	norm_mat.row(2).col(1) = 0;
	norm_mat.row(2).col(2) = 1;

	X_norm = norm_mat * X;
}
float Residual(Mat H, Mat X1, Mat X2)
{
	int num = X1.cols;
	//评价误差
	Mat X2_ = H*X1;
	Mat X2_row_3 = Mat::zeros(3, num, CV_32F);
	X2_.row(2).copyTo(X2_row_3.row(0));
	X2_.row(2).copyTo(X2_row_3.row(1));
	X2_.row(2).copyTo(X2_row_3.row(2));
	X2_ /= X2_row_3;
	Mat dx = X2_.row(0) - X2.row(0);
	Mat dy = X2_.row(1) - X2.row(1);
	Mat d_x_y = (dx.mul(dx) + dy.mul(dy));
	//返回值err
	float err = sum(d_x_y).val[0];

	return err;
}
//快排
void quick_sort(float *s, vector<Mat> &H_list, int l, int r)
{  
	if (l < r)
	{
		int i = l, j = r;
		float x = s[l];
		Mat temp = H_list[l].clone();
		while (i < j)
		{
			while(i < j && s[j] >= x) // 从右向左找第一个小于x的数  
				j--;
			if(i < j)
			{
				s[i++] = s[j];
				H_list[j].copyTo(H_list[i-1]);
			}

			while(i < j && s[i] < x) // 从左向右找第一个大于等于x的数  
				i++;
			if(i < j)
			{
				s[j--] = s[i];
				H_list[i].copyTo(H_list[j+1]);
			}
		}
		s[i] = x;
		temp.copyTo(H_list[i]);
		quick_sort(s, H_list, l, i - 1); // 递归调用   
		quick_sort(s, H_list, i + 1, r);
	}
}
//参数限定
void constrain_coefficients(Mat &H)
{
	//倾斜参数限定
	if(((float*)H.data)[1] > 0.04)
		((float*)H.data)[1] = 0.04;
	else if(((float*)H.data)[1] < -0.04)
		((float*)H.data)[1] = -0.04;
	if(((float*)H.data)[3] > 0.04)
		((float*)H.data)[3] = 0.04;
	else if(((float*)H.data)[3] < -0.04)
		((float*)H.data)[3] = -0.04;
	//放缩参数限定
	if(((float*)H.data)[0] > 1.04)
		((float*)H.data)[0] = 1.04;
	else if(((float*)H.data)[0] < 0.96)
		((float*)H.data)[0] = 0.96;
	if(((float*)H.data)[4] > 1.04)
		((float*)H.data)[4] = 1.04;
	else if(((float*)H.data)[4] < 0.96)
		((float*)H.data)[4] = 0.96;
}
//参数检查
bool check_coefficients(Mat &H)
{
	//降采样2倍
	//倾斜、放缩参数检查
	if((((float*)H.data)[1] > 0.1) || (((float*)H.data)[1] < -0.1) || (((float*)H.data)[3] > 0.1) || (((float*)H.data)[3] < -0.1) || \
		(((float*)H.data)[0] > 1.1) || (((float*)H.data)[0] < 0.9) || (((float*)H.data)[4] > 1.1) || (((float*)H.data)[4] < 0.9))
		return false;
	else
		return true;
}
void Nelder_Mead(Mat &H0, Mat &pt_bg_inlier, Mat &cor_smooth_inlier, int max_iter, float eps, Mat &H, bool show_best)
{
	const int Max_time = max_iter;
	//透视变换矩阵，即单应性矩阵时，有8个变量
	int var_num = 8;
	vector<Mat> vx(var_num+1);
	H0.copyTo(vx[0]);
	float vf[9] = {0, 0, 0, 0, 0, 0, 0};
	vf[0] = Residual(H0, pt_bg_inlier, cor_smooth_inlier);
	//只将单应矩阵的前两行代入计算
	//cout<<H0<<endl;
	for(int i=0; i<3; i++)
	{
		for(int j=0; j<3; j++)
		{
			if(!(i==2 && j==2))
			{
				H0.copyTo(vx[i*3+j+1]);
				if((fabs(((float*)H0.data)[i*3+j])) < 0.00005)	//如果太小，则认为加上一个很小的扰动
					((float*)vx[i*3+j+1].data)[i*3+j] += 0.005;
				else
					((float*)vx[i*3+j+1].data)[i*3+j] /= 1.05;		//否则，乘以一个系数
				//参数限定
				//constrain_coefficients(vx[i*3+j+1]);
				//cout<<vx[i*3+j+1]<<endl;
				vf[i*3+j+1] = Residual(vx[i*3+j+1], pt_bg_inlier, cor_smooth_inlier);	//计算其对应误差
			}
		}
	}
	//排序
	quick_sort(vf, vx, 0, var_num);

	float max_of_this = 0;
	float max_err = 0;
	while(max_iter>0)
	{
		for(int i=0; i<var_num+1; i++)
		{
			for(int j=i+1; j<var_num+1; j++)
			{
				Mat abs_err = abs(vx[i] - vx[j]);
				for(int k=0; k<3; k++)
				{
					if(((float*)abs_err.data)[k] > max_of_this)
						max_of_this = ((float*)abs_err.data)[k];
					if(((float*)abs_err.data)[k+3] > max_of_this)
						max_of_this = ((float*)abs_err.data)[k+3];
				}
				if(max_of_this > max_err)
					max_err = max_of_this;
			}
		}
		//max_err = fabs(vf[0] - vf[var_num]);
		if(show_best && max_iter %100 == 0)
		{
			if(max_iter %100 == 0)
				cout<<max_err<<"\t";
		}
		//如果各个参数的最大误差足够小，则跳出循环
		//有时候，轨迹数比较少，2*pt_bg_inlier.cols就比较小，收敛条件就会太苛刻
		if(max_err < eps && (vf[0] <= 2*pt_bg_inlier.cols || vf[0] <= 120))
		{
			if(show_best)
			{
				cout<<"迭代次数:"<<Max_time-max_iter<<endl;
				cout<<"最大最小相差为:"<<max_err<<endl;
				cout<<"已找到最优结果，最小相差为:"<<vf[0]<<endl;
			}
			break;
		}
		//算法核心模块
		Mat best = vx[0];
		float fbest = vf[0];
		Mat soso = vx[var_num-1];
		float fsoso = vf[var_num-1];
		Mat worst = vx[var_num];
		float fworst = vf[var_num];
		Mat center = Mat::zeros(3, 3, CV_32F);
		for(int i=0; i<var_num; i++)
			center += vx[i];
		center /= var_num;
		Mat r = 2*center - worst;
		//参数限定
		//constrain_coefficients(r);
		float fr = Residual(r, pt_bg_inlier, cor_smooth_inlier);
		if(fr < fbest)
		{
			//比最好的结果还好，说明方向正确，考察扩展点，以期望更多的下降
			Mat e = 2*r - center;
			//参数限定
			//constrain_coefficients(e);
			float fe = Residual(e, pt_bg_inlier, cor_smooth_inlier);
			//在扩展点和反射点中选择较优者去替换最差点
			if(fe < fr)
			{
				vx[var_num] = e;//e.clone();
				vf[var_num] = fe;
			}
			else
			{
				vx[var_num] = r;//r.clone();
				vf[var_num] = fr;
			}
		}
		else
		{
			if(fr < fsoso)
			{
				//比次差结果好，能改进
				vx[var_num] = r;//r.clone();
				vf[var_num] = fr;
			}
			else//比次差结果还差，应考虑压缩点
			{
				//当压缩点无法得到更优值的时候，考虑收缩
				bool shrink = false;
				if(fr < fworst)
				{
					//由于r点更优，所以向r点的方向找压缩点
					Mat c = (r + center)/2;
					//参数限定
					//constrain_coefficients(c);
					float fc = Residual(c, pt_bg_inlier, cor_smooth_inlier);
					if(fc < fr)
					{
						//确定从r压缩向c可以改进
						vx[var_num] = c;//c.clone();
						vf[var_num] = fc;
					}
					else
						//否则的话，准备进行收缩
						shrink = true;
				}
				else
				{
					//由于w点更优，所以向w点的方向找压缩点
					Mat c = (worst + center)/2;
					//参数限定
					//constrain_coefficients(c);
					float fc = Residual(c, pt_bg_inlier, cor_smooth_inlier);
					if(fc < fr)
					{
						//确定从r压缩向c可以改进
						vx[var_num] = c;//c.clone();
						vf[var_num] = fc;
					}
					else
						//否则的话，准备进行收缩
						shrink = true;
				}
				if(shrink)
				{
					for(int i=1; i<var_num+1; i++)
					{
						Mat temp = (vx[i] + best) / 2;
						//参数限定
						//constrain_coefficients(temp);
						vx[i] = temp;//temp.clone();
						vf[i] = Residual(vx[i], pt_bg_inlier, cor_smooth_inlier);
					}
				}
			}
		}
		//排序
		quick_sort(vf, vx, 0, var_num);
		//if(max_iter>900)
		//	cout<<"最小误差是"<<vf[0]<<endl;
		max_iter--;
	}
	H = vx[0].clone();
	//cout<<"最优结果"<<H<<endl;
	//cout<<"最小误差是"<<vf[0]<<endl;
	//cout<<Residual(H0, pt_bg_inlier, cor_smooth_inlier);
}
inline float cross_product(Point2f &A, Point2f &B, Point2f &C)
{
	return ((A.x-C.x)*(B.y-C.y)-(B.x-C.x)*(A.y-C.y));
}
//计算单应矩阵，替换findHomography函数,max_iter为Nelder-Mead算法最大迭代次数
void RANSAC_Foreground_Judgement(vector<Point2f> &pt_bg_cur, vector<Point2f> &Trj_cor_smooth, int max_iter, bool show_ransac, float scale, unsigned char *Foreground_times, int width, int height)
{
	//int64 st, et;
	//st = cvGetTickCount();
	int RANSAC_times = 100;
	float thresh_inlier = 80;//80/(scale*scale);
	int num = pt_bg_cur.size();
	//构造归一化坐标向量的矩阵
	Mat pt_bg = Mat::ones(3, num, CV_32F), cor_smooth = Mat::ones(3, num, CV_32F);
	//用于产生随机序列
	vector<int>index_shuffle(num);
	vector<int>block_of_point(num);		//记录每个点属于哪个块
	vector<int>block_num(6);						//记录每个块内的点数
	int blocks = 0;										//记录实际几个块有点
	int row_n = 0, col_n = 0;
	for(int i=0; i<num; i++)
	{
		index_shuffle[i] = i;
		((float*)pt_bg.data)[i] = pt_bg_cur[i].x;
		((float*)pt_bg.data)[i+num] = pt_bg_cur[i].y;
		((float*)cor_smooth.data)[i] = Trj_cor_smooth[i].x;
		((float*)cor_smooth.data)[i+num] = Trj_cor_smooth[i].y;
		row_n = ((int)pt_bg_cur[i].y)/((int)(height/2));
		col_n = ((int)pt_bg_cur[i].x)/((int)(width/3));
		block_of_point[i] = 3*row_n + col_n;
		block_num[block_of_point[i]]++;
	}
	for (int i=0; i<6; i++)
		if (block_num[i])
			blocks++;
	//ofstream outfile_2("temp_smooth.txt");
	//outfile_2<<cor_smooth<<endl;
	//ofstream outfile_1("temp_shaky.txt");
	//outfile_1<<pt_bg<<endl;
	//RANSAC算法，最多100次循环
	srand((unsigned)time(0));
	Mat OK = Mat::zeros(RANSAC_times, num, CV_8U);			//好的结果，1表示该数据与模型匹配得好，0为不好
	vector<int> Score(RANSAC_times);								//评价误差得分，得分越高表示模型越好
	vector<Mat> H(RANSAC_times);									//每次的单应矩阵
	vector<float> Total_err(RANSAC_times);						//总体误差
	Mat thresh = thresh_inlier*Mat::ones(1, num, CV_32F);
	int best_index = -1;		//最好模型的索引值
	int best = -1;				//Score的最大值
	//搜不到在合适范围内的最优值，就再循环一次
	while(best == -1)
	{
		for(int t=0; t<RANSAC_times; t++)
		{
			//随机抽取四个点，构造左边A矩阵
			vector<int> rand_set;
			vector<int>block_has(6);						//记录在每次RANSAC过程中每个块已经进去的点数，限制每个块最多2个点
			//先用shuffle算法生成随机序列
			random_shuffle(index_shuffle.begin(), index_shuffle.end());
			//限制每个块最多有两个点进入RANSAC，前两个点的压入不用判定
			rand_set.push_back(index_shuffle[0]);
			block_has[block_of_point[index_shuffle[0]]]++;
			rand_set.push_back(index_shuffle[1]);
			block_has[block_of_point[index_shuffle[1]]]++;
			if (blocks==1)
			{
				rand_set.push_back(index_shuffle[2]);
				rand_set.push_back(index_shuffle[3]);
			}
			else
			{
				//压入第三个点
				int shuffle_k = 0;
				int shuffle_index = 2;			
				shuffle_k = index_shuffle[shuffle_index];
				while (block_has[block_of_point[shuffle_k]] == 2 && shuffle_index<num-2)
				{
					shuffle_index++;
					shuffle_k = index_shuffle[shuffle_index];
				}
				block_has[block_of_point[shuffle_k]]++;
				rand_set.push_back(index_shuffle[shuffle_index]);

				//压入第四个点
				shuffle_index++;
				shuffle_k = index_shuffle[shuffle_index];
				while (block_has[block_of_point[shuffle_k]] == 2 && shuffle_index<num-1)
				{
					shuffle_index++;
					shuffle_k = index_shuffle[shuffle_index];
				}
				block_has[block_of_point[shuffle_k]]++;
				rand_set.push_back(index_shuffle[shuffle_index]);
			}

			//A一定要设定为局部变量！！！因为下面使用+=，而不是赋值=！！！！
			Mat A = Mat::zeros(12, 9, CV_32F);
			int j = 0;
			int k = rand_set[0];	//0 <= k < num
			Mat hat = (Mat_<float>(3,3) << 0, -1, ((float*)cor_smooth.data)[k+num], 1, 0, -1*((float*)cor_smooth.data)[k], -1*((float*)cor_smooth.data)[k+num], ((float*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			float x = ((float*)pt_bg.data)[k];
			float y = ((float*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			++j;
			k = rand_set[1];	//0 <= k < num
			hat = (Mat_<float>(3,3) << 0, -1, ((float*)cor_smooth.data)[k+num], 1, 0, -1*((float*)cor_smooth.data)[k], -1*((float*)cor_smooth.data)[k+num], ((float*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			x = ((float*)pt_bg.data)[k];
			y = ((float*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			++j;
			k = rand_set[2];	//0 <= k < num
			hat = (Mat_<float>(3,3) << 0, -1, ((float*)cor_smooth.data)[k+num], 1, 0, -1*((float*)cor_smooth.data)[k], -1*((float*)cor_smooth.data)[k+num], ((float*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			x = ((float*)pt_bg.data)[k];
			y = ((float*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			++j;
			k = rand_set[3];	//0 <= k < num
			hat = (Mat_<float>(3,3) << 0, -1, ((float*)cor_smooth.data)[k+num], 1, 0, -1*((float*)cor_smooth.data)[k], -1*((float*)cor_smooth.data)[k+num], ((float*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			x = ((float*)pt_bg.data)[k];
			y = ((float*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			//cout<<A<<endl;
			//SVD分解生成VT，第9行为最小特征值对应的特征向量
			SVD thissvd(A,SVD::FULL_UV);
			Mat VT=thissvd.vt; 
			//cout<<VT<<endl;
			//生成本次RANSAC循环对应的归一化的单应矩阵
			H[t] = (Mat_<float>(3,3) <<((float*)VT.data)[72], ((float*)VT.data)[75], ((float*)VT.data)[78], ((float*)VT.data)[73], ((float*)VT.data)[76], ((float*)VT.data)[79], ((float*)VT.data)[74], ((float*)VT.data)[77], ((float*)VT.data)[80]);// / ((float*)VT.data)[80];
			//cout<<H[t]<<endl;
			H[t] /= ((float*)H[t].data)[8];
			//cout<<H[t]<<endl;

			//评价误差
			Mat X2_ = H[t]*pt_bg;
			//cout<<X2_<<endl;
			Mat X2_row_3 = Mat::zeros(3, num, CV_32F);
			X2_.row(2).copyTo(X2_row_3.row(0));
			X2_.row(2).copyTo(X2_row_3.row(1));
			X2_.row(2).copyTo(X2_row_3.row(2));
			X2_ /= X2_row_3;
			//cout<<X2_<<endl;
			Mat dx = X2_.row(0) - cor_smooth.row(0);
			Mat dy = X2_.row(1) - cor_smooth.row(1);
			Mat d_x_y = (dx.mul(dx) + dy.mul(dy));
			//cout<<d_x_y<<endl;
			//结果记录在Total_err、OK和Score矩阵中
			Total_err[t] = sum(d_x_y).val[0];
			OK.row(t) = (d_x_y < thresh) / 255.f;
			//cout<<OK.row(t)<<endl;
			Scalar sum_o = sum(OK.row(t));
			Score[t] = sum(OK.row(t)).val[0];
			//记录最好结果的索引值
			if(Score[t] > best)
			{
				//cout<<H[t]<<endl;
				if(check_coefficients(H[t]))
				{
					best = Score[t];
					best_index = t;
				}
			}
			else if(Score[t] == best)	//模型匹配数量一致时，取误差最小的
			{
				if(Total_err[t] < Total_err[best_index])
				{
					if(check_coefficients(H[t]))
					{
						best = Score[t];
						best_index = t;
					}
				}
			}
		}
	}
	//et = cvGetTickCount();
	//printf("RANSAC循环，100次时间为: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);
	if(show_ransac)
	{
		cout<<"最好的模型是第"<<best_index<<"个"<<endl;
		cout<<"匹配上了"<<best<<"个"<<endl;
		cout<<"最好的模型是:\n"<<H[best_index]<<endl;
	}
	Mat ok = OK.row(best_index);
	for (int i=0; i<num; i++)
		if (!ok.data[i])
			Foreground_times[i]++;

}

//计算单应矩阵，替换findHomography函数,max_iter为Nelder-Mead算法最大迭代次数
Mat Homography_Nelder_Mead_with_outliers(vector<Point2f> &pt_bg_cur, vector<Point2f> &Trj_cor_smooth, int max_iter, Mat& outliers, int height)
{
	//int64 st, et;
	//st = cvGetTickCount();
	int RANSAC_times = 200;
	float thresh_inlier = 30;//80/(scale*scale);
	int num = pt_bg_cur.size();
	//构造归一化坐标向量的矩阵
	Mat pt_bg = Mat::ones(3, num, CV_32F), cor_smooth = Mat::ones(3, num, CV_32F);
	//用于产生随机序列
	vector<int>index_shuffle(num);
	for(int i=0; i<num; i++)
	{
		index_shuffle[i] = i;
		((float*)pt_bg.data)[i] = pt_bg_cur[i].x;
		((float*)pt_bg.data)[i+num] = pt_bg_cur[i].y;
		((float*)cor_smooth.data)[i] = Trj_cor_smooth[i].x;
		((float*)cor_smooth.data)[i+num] = Trj_cor_smooth[i].y;
	}

	//RANSAC算法，最多100次循环
	srand((unsigned)time(0));
	Mat OK = Mat::zeros(RANSAC_times, num, CV_8U);			//好的结果，1表示该数据与模型匹配得好，0为不好
	vector<int> Score(RANSAC_times);								//评价误差得分，得分越高表示模型越好
	vector<Mat> H(RANSAC_times);									//每次的单应矩阵
	vector<float> Total_err(RANSAC_times);						//总体误差
	Mat thresh = thresh_inlier*Mat::ones(1, num, CV_32F);
	Mat every_outliers = Mat::zeros(RANSAC_times, num, CV_8U);
	int best_index = -1;		//最好模型的索引值
	int best = -1;				//Score的最大值
	//搜不到在合适范围内的最优值，就再循环一次
	while(best == -1)
	{
		for(int t=0; t<RANSAC_times; t++)
		{
			//随机抽取四个点，构造左边A矩阵
			vector<int> rand_set;
			//先用shuffle算法生成随机序列
			random_shuffle(index_shuffle.begin(), index_shuffle.end());
			rand_set.push_back(index_shuffle[0]);
			rand_set.push_back(index_shuffle[1]);
			rand_set.push_back(index_shuffle[2]);
			rand_set.push_back(index_shuffle[3]);

			//A一定要设定为局部变量！！！因为下面使用+=，而不是赋值=！！！！
			Mat A = Mat::zeros(12, 9, CV_32F);
			int j = 0;
			int k = rand_set[0];	//0 <= k < num
			Mat hat = (Mat_<float>(3,3) << 0, -1, ((float*)cor_smooth.data)[k+num], 1, 0, -1*((float*)cor_smooth.data)[k], -1*((float*)cor_smooth.data)[k+num], ((float*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			float x = ((float*)pt_bg.data)[k];
			float y = ((float*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			++j;
			k = rand_set[1];	//0 <= k < num
			hat = (Mat_<float>(3,3) << 0, -1, ((float*)cor_smooth.data)[k+num], 1, 0, -1*((float*)cor_smooth.data)[k], -1*((float*)cor_smooth.data)[k+num], ((float*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			x = ((float*)pt_bg.data)[k];
			y = ((float*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			++j;
			k = rand_set[2];	//0 <= k < num
			hat = (Mat_<float>(3,3) << 0, -1, ((float*)cor_smooth.data)[k+num], 1, 0, -1*((float*)cor_smooth.data)[k], -1*((float*)cor_smooth.data)[k+num], ((float*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			x = ((float*)pt_bg.data)[k];
			y = ((float*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			++j;
			k = rand_set[3];	//0 <= k < num
			hat = (Mat_<float>(3,3) << 0, -1, ((float*)cor_smooth.data)[k+num], 1, 0, -1*((float*)cor_smooth.data)[k], -1*((float*)cor_smooth.data)[k+num], ((float*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			x = ((float*)pt_bg.data)[k];
			y = ((float*)pt_bg.data)[k+num];
			A.rowRange(j*3, j*3+3).colRange(0,3) += hat*x;
			A.rowRange(j*3, j*3+3).colRange(3,6) += hat*y;
			A.rowRange(j*3, j*3+3).colRange(6,9) += hat;

			//cout<<A<<endl;
			//ofstream A_file("A_file.txt");
			//A_file<<A<<endl;
			//SVD分解生成VT，第9行为最小特征值对应的特征向量
			SVD thissvd(A,SVD::FULL_UV);
			Mat VT=thissvd.vt; 
			//cout<<VT<<endl;
			//生成本次RANSAC循环对应的归一化的单应矩阵
			H[t] = (Mat_<float>(3,3) <<((float*)VT.data)[72], ((float*)VT.data)[75], ((float*)VT.data)[78], ((float*)VT.data)[73], ((float*)VT.data)[76], ((float*)VT.data)[79], ((float*)VT.data)[74], ((float*)VT.data)[77], ((float*)VT.data)[80]);// / ((float*)VT.data)[80];
			//cout<<H[t]<<endl;
			H[t] /= ((float*)H[t].data)[8];
			//cout<<H[t]<<endl;

			//评价误差
			Mat X2_ = H[t]*pt_bg;
			//cout<<X2_<<endl;
			Mat X2_row_3 = Mat::zeros(3, num, CV_32F);
			X2_.row(2).copyTo(X2_row_3.row(0));
			X2_.row(2).copyTo(X2_row_3.row(1));
			X2_.row(2).copyTo(X2_row_3.row(2));
			X2_ /= X2_row_3;
			//cout<<X2_<<endl;
			Mat dx = X2_.row(0) - cor_smooth.row(0);
			Mat dy = X2_.row(1) - cor_smooth.row(1);
			Mat d_x_y = (dx.mul(dx) + dy.mul(dy));
			//cout<<d_x_y<<endl;
			//结果记录在Total_err、OK和Score矩阵中
			Total_err[t] = sum(d_x_y).val[0];
			OK.row(t) = (d_x_y < thresh) / 255.f;
			//cout<<OK.row(t)<<endl;
			Scalar sum_o = sum(OK.row(t));
			Score[t] = sum(OK.row(t)).val[0];
			//记录最好结果的索引值
			if(Score[t] > best)
			{
				//cout<<H[t]<<endl;
				if(check_coefficients(H[t]))
				{
					best = Score[t];
					best_index = t;
				}
			}
			else if(Score[t] == best)	//模型匹配数量一致时，取误差最小的
			{
				if(Total_err[t] < Total_err[best_index])
				{
					if(check_coefficients(H[t]))
					{
						best = Score[t];
						best_index = t;
					}
				}
			}
		}
	}
	//et = cvGetTickCount();
	//printf("RANSAC循环，100次时间为: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);
	//cout<<"匹配上了"<<best<<"个"<<endl;
	//outliers = OK.row(best_index).clone();
	//cout<<outliers<<endl;
	//提取出内点
	Mat pt_bg_inlier = Mat::zeros(3, best, CV_32F), cor_smooth_inlier = Mat::zeros(3, best, CV_32F);
	int inlier_ind = 0;
	//cout<<"OK.row(best_index)"<<endl;
	//cout<<OK.row(best_index)<<endl;
	for(int i=0; i<num; i++)
	{
		if(((unsigned char*)OK.data)[best_index*num+i] > 0)
		{
			pt_bg.col(i).copyTo(pt_bg_inlier.col(inlier_ind));
			cor_smooth.col(i).copyTo(cor_smooth_inlier.col(inlier_ind));
			inlier_ind++;
		}
	}
	//Nelder-Mead算法搜索最优值
	//强制单应矩阵变为仿射矩阵，即第3行前两个元素为0
	Mat H0 = H[best_index];
	//Mat H_best = Mat::zeros(3, 3, CV_32F);
	//H0.row(2).col(0) = 0.f;
	//H0.row(2).col(1) = 0.f;
	Mat H_NM = Mat::zeros(3, 3, CV_32F);
	float eps = 0.05;
	//st = cvGetTickCount();
	bool show_ransac = false;
	Nelder_Mead(H0, pt_bg_inlier, cor_smooth_inlier, max_iter, eps, H_NM, show_ransac);
	//et = cvGetTickCount();
	//printf("NM搜索时间: %f\n", (et-st)/(float)cvGetTickFrequency()/1000.);
	return H_NM;
}
