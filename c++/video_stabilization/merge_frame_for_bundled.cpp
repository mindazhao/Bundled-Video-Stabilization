/*
2018.11.16 修改了const int max_dist = 20000;//这个数据库比较特殊，差的很多，所以要设的比较大，之前特别不准，导致feature_loss 这一项很大，达到了0.15
*/



#include <opencv2/opencv.hpp>
#include <iostream>
#include<io.h>
#include<direct.h>
using namespace cv;
using namespace std;


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