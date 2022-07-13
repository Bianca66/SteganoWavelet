//Declaration of filter functions

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgcodecs.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <stdint.h>
#include <math.h>
#include <mpi.h>

using namespace std;
using namespace cv;

float* picture_m;
float* picture_m_c;
uint8_t* arr_hide;

class Image {

private:
	cv::Mat pic;
	int colors;

public:

	Image(cv::Mat pic, int colors) {
		this->pic = pic.clone();
		this->colors = colors;
	}

	int getWidth() {
		return pic.rows;
	}

	int getHeight() {
		return pic.cols;
	}

	int getColors() {
		return colors;
	}

	cv::Mat getPic() {
		return pic;
	}

	void display() {
		namedWindow("Display frame", WINDOW_NORMAL);
		cv::imshow("Display frame", getPic());
	}

	void check() {
		if (getPic().empty())
			cout << "Fail to open!" << endl;
		else
			cout << "Image opened fine!" << endl;
	}
};

Image applyWavelet(Image);
Image hideImage(Image, Image);
void extractImage(Image, Image);

int main(int argc, char** argv) {

	// read the initial image
	cv::Mat pic = cv::imread("flower.bmp", 1);
	// create the input image object
	Image img(pic, 3);
	// check the image
	img.check();
	// display the image
	//img.display();
	// wait key
	cv::waitKey();
	// get the dimension of the image
	int h = img.getHeight();
	int w = img.getWidth();
	int c = img.getColors();
	// display dimension
	cout << "Input image size: " << h << " x " << w << endl;
	// allocate dynamic memory to store the image
	picture_m = new float[h * w * c];
	picture_m_c = new float[h * w * c];
	// return the image after the wavelet transform
	Image waveletImg = applyWavelet(img);
	// display wavelet image
	//waveletImg.display();
	// wait key
	cv::waitKey();
	// read the image to be hidden
	Mat hiddenPic = imread("lena.bmp", 1);
	// create hidden image object
	Image hiddenImg(hiddenPic, 3);
	// display the image to be hidden
	//hiddenImg.display();
	// wait key
	cv::waitKey();
	// allocate dynamic memory for the hidden arrays
	arr_hide = new uint8_t[4 * hiddenImg.getHeight() * hiddenImg.getWidth() * c];
	Image doneImg = hideImage(hiddenImg, waveletImg);
	// display the final image
	//doneImg.display();
	// wait key
	cv::waitKey();
	// write final image into file
	imwrite("flower_hidden.bmp", hiddenImg.getPic());

	// extract image
	//extractImage(doneImg, hiddenImg);

	// free the allocated space
	delete[] picture_m;
	delete[] picture_m_c;
	delete[] arr_hide;


	// Finalize the MPI environment.
	MPI_Finalize();


	return 0;
}

Image applyWavelet(Image pic) {
	// copy the original image
	Mat original = pic.getPic();

	// get the dimension of the image
	int h = pic.getHeight();
	int w = pic.getWidth();
	int c = pic.getColors();

	// store the input image into picture_m dynamic array
	// and apply the corresponding bias value for each pixel
	for (int k = 0; k < c; k++) {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				if (original.at<cv::Vec3b>(i, j)[k] > 128)
					picture_m[h * w * k + w * i + j] = original.at<cv::Vec3b>(i, j)[k] - 5;
				else
					picture_m[h * w * k + w * i + j] = original.at<cv::Vec3b>(i, j)[k] + 5;
			}
		}
	}

	// column transform
	for (int k = 0; k < c; k++) {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w / 2; j++) {
				picture_m_c[h * w * k + w * i + j] = (picture_m[h * w * k + w * i + 2 * j] + picture_m[h * w * k + w * i + 2 * j + 1]) / 2;
				picture_m_c[h * w * k + w * i + j + w / 2] = (picture_m[h * w * k + w * i + 2 * j] - picture_m[h * w * k + w * i + 2 * j + 1]) / 2;
			}
		}
	}

	// copy the column-transformed image
	// into the picture_m dynamic array
	for (int k = 0; k < c; k++) {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				picture_m[h * w * k + w * i + j] = picture_m_c[h * w * k + h * i + j];
			}
		}
	}


	// execute row transform only
	// for the 1st half of the image
	for (int k = 0; k < c; k++) {
		for (int j = 0; j < w / 2; j++) {
			for (int i = 0; i < h / 2; i++) {
				picture_m[h * w * k + h * i + j] = (picture_m_c[h * w * k + 2 * h * i + j] + picture_m_c[h * w * k + 2 * h * i + w + j]) / 2;
				picture_m[h * w * k + h * (i + h / 2) + j] = (picture_m_c[h * w * k + 2 * h * i + j] - picture_m_c[h * w * k + 2 * h * i + w + j]) / 2;
			}
		}
	}

	// row reverse-transform
	for (int k = 0; k < c; k++) {
		for (int j = 0; j < w / 2; j++) {
			for (int i = 0; i < h / 2; i++) {
				picture_m_c[h * w * k + h * (2 * i) + j] = picture_m[h * w * k + h * i + j] + picture_m[h * w * k + h * (i + w / 2) + j];
				picture_m_c[h * w * k + h * (2 * i + 1) + j] = picture_m[h * w * k + h * i + j] - picture_m[h * w * k + h * (i + w / 2) + j];
			}
		}
	}


	// copy the half-row-transformed image
	// into the picture_m dynamic array and round the result

	for (int k = 0; k < c; k++) {
		for (int j = 0; j < w; j++) {
			for (int i = 0; i < h; i++) {
				picture_m[h * w * k + h * i + j] = round(picture_m[h * w * k + h * i + j]);
			}
		}
	}

	for (int k = 0; k < c; k++) {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				original.at<cv::Vec3b>(i, j)[k] = picture_m[h * w * k + w * i + j];
			}
		}
	}

	Image waveletImg(original, c);

	return waveletImg;
}

Image hideImage(Image hiddenPic, Image waveletPic) {
	// copy the hidden image
	Mat hidden = hiddenPic.getPic();

	// get the dimension of the hidden image
	int hh = hiddenPic.getHeight();
	int hw = hiddenPic.getWidth();
	int hc = hiddenPic.getColors();

	// copy the wavelet image
	Mat wavelet = waveletPic.getPic();

	// get the dimension of the wavelet image
	int h = waveletPic.getHeight();
	int w = waveletPic.getWidth();
	int c = waveletPic.getColors();

	// create the arrays to be hidden
	for (int k = 0; k < c; k++) {
		for (int i = 0; i < hh; i++) {
			for (int j = 0; j < hw; j++) {
				arr_hide[hh * hw * 4 * k + w * i + 4 * j] = (hidden.at<cv::Vec3b>(i, j)[k] & 0xc0) >> 6;
				arr_hide[hh * hw * 4 * k + w * i + 4 * j + 1] = (hidden.at<cv::Vec3b>(i, j)[k] & 0x30) >> 4;
				arr_hide[hh * hw * 4 * k + w * i + 4 * j + 2] = (hidden.at<cv::Vec3b>(i, j)[k] & 0x0c) >> 2;
				arr_hide[hh * hw * 4 * k + w * i + 4 * j + 3] = (hidden.at<cv::Vec3b>(i, j)[k] & 0x03);
			}
		}
	}

	// hide the data inside the input image
	for (int k = 0; k < c; k++) {
		for (int i = 0; i < hh; i++) {
			for (int j = 0; j < hw; j++) {
				if (picture_m[h * w * k + w * i + 4 * j] < 0)
					picture_m[h * w * k + w * i + 4 * j] = -(((int)(-picture_m[h * w * k + w * i + 4 * j]) & 0xfc) + arr_hide[hh * hw * 4 * k + w * i + 4 * j]);
				else
					picture_m[h * w * k + w * i + 4 * j] = ((int)(picture_m[h * w * k + w * i + 4 * j]) & 0xfc) + arr_hide[hh * hw * 4 * k + w * i + 4 * j];

				if (picture_m[h * w * k + w * i + 4 * j + 1] < 0)
					picture_m[h * w * k + w * i + 4 * j + 1] = -(((int)(-picture_m[h * w * k + w * i + 4 * j + 1]) & 0xfc) + arr_hide[hh * hw * 4 * k + w * i + 4 * j + 1]);
				else
					picture_m[h * w * k + w * i + 4 * j + 1] = ((int)(picture_m[h * w * k + w * i + 4 * j + 1]) & 0xfc) + arr_hide[hh * hw * 4 * k + w * i + 4 * j + 1];

				if (picture_m[h * w * k + w * i + 4 * j + 2] < 0)
					picture_m[h * w * k + w * i + 4 * j + 2] = -(((int)(-picture_m[h * w * k + w * i + 4 * j + 2]) & 0xfc) + arr_hide[hh * hw * 4 * k + w * i + 4 * j + 2]);
				else
					picture_m[h * w * k + w * i + 4 * j + 2] = ((int)(picture_m[h * w * k + w * i + 4 * j + 2]) & 0xfc) + arr_hide[hh * hw * 4 * k + w * i + 4 * j + 2];

				if (picture_m[h * w * k + w * i + 4 * j + 3] < 0)
					picture_m[h * w * k + w * i + 4 * j + 3] = -(((int)(-picture_m[h * w * k + w * i + 4 * j + 3]) & 0xfc) + arr_hide[hh * hw * 4 * k + w * i + 4 * j + 3]);
				else
					picture_m[h * w * k + w * i + 4 * j + 3] = ((int)(picture_m[h * w * k + w * i + 4 * j + 3]) & 0xfc) + arr_hide[hh * hw * 4 * k + w * i + 4 * j + 3];
			}
		}
	}

	// copy the transformed pixels
	for (int k = 0; k < c; k++) {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				picture_m_c[h * w * k + w * i + j] = picture_m[h * w * k + w * i + j];
			}
		}
	}

	// row reverse-transform
	for (int k = 0; k < c; k++) {
		for (int j = 0; j < w / 2; j++) {
			for (int i = 0; i < h / 2; i++) {
				picture_m_c[h * w * k + h * (2 * i) + j] = picture_m[h * w * k + h * i + j] + picture_m[h * w * k + h * (i + w / 2) + j];
				picture_m_c[h * w * k + h * (2 * i + 1) + j] = picture_m[h * w * k + h * i + j] - picture_m[h * w * k + h * (i + w / 2) + j];
			}
		}
	}

	for (int k = 0; k < c; k++) {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				picture_m[h * w * k + w * i + j] = picture_m_c[h * w * k + w * i + j];
			}
		}
	}

	for (int k = 0; k < c; k++) {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w / 2; j++) {
				picture_m[h * w * k + w * i + 2 * j] = picture_m_c[h * w * k + w * i + j] + picture_m_c[h * w * k + w * i + j + w / 2];
				picture_m[h * w * k + w * i + 2 * j + 1] = picture_m_c[h * w * k + w * i + j] - picture_m_c[h * w * k + w * i + j + w / 2];
			}
		}
	}

	Mat custom = waveletPic.getPic();

	for (int k = 0; k < c; k++) {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				custom.at<cv::Vec3b>(i, j)[k] = picture_m[h * w * k + w * i + j];
			}
		}
	}

	Image hiddenImg(custom, hc);

	//hiddenImg.display();

	return hiddenImg;
}

void extractImage(Image finalImg, Image hiddenImg) {

	// copy the hidden image
	Mat final = finalImg.getPic();

	// get the dimension of the hidden image
	int h = finalImg.getHeight();
	int w = finalImg.getWidth();
	int c = finalImg.getColors();

	for (int k = 0; k < c; k++) {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				picture_m[h * w * k + w * i + j] = final.at<cv::Vec3b>(i, j)[k];
			}
		}
	}

	// column transform
	for (int k = 0; k < c; k++) {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w / 2; j++) {
				picture_m_c[h * w * k + w * i + j] = (picture_m[h * w * k + w * i + 2 * j] + picture_m[h * w * k + w * i + 2 * j + 1]) / 2;
				picture_m_c[h * w * k + w * i + j + w / 2] = (picture_m[h * w * k + w * i + 2 * j] - picture_m[h * w * k + w * i + 2 * j + 1]) / 2;
			}
		}
	}

	// copy the transformed pixels
	for (int k = 0; k < c; k++) {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				picture_m[h * w * k + w * i + j] = picture_m_c[h * w * k + w * i + j];
			}
		}
	}

	// execute row transform only
	// for the 1st half of the image
	for (int k = 0; k < c; k++) {
		for (int j = 0; j < w / 2; j++) {
			for (int i = 0; i < h / 2; i++) {
				picture_m[h * w * k + h * i + j] = (picture_m_c[h * w * k + 2 * h * i + j] + picture_m_c[h * w * k + 2 * h * i + w + j]) / 2;
				picture_m[h * w * k + h * (i + h / 2) + j] = (picture_m_c[h * w * k + 2 * h * i + j] - picture_m_c[h * w * k + 2 * h * i + w + j]) / 2;
			}
		}
	}

	Mat custom = finalImg.getPic();

	for (int k = 0; k < c; k++) {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				custom.at<cv::Vec3b>(i, j)[k] = picture_m[h * w * k + w * i + j];
			}
		}
	}

	Image hiddenWavelet(custom, c);
	//hiddenWavelet.display();
	cv::waitKey();

	// copy the hidden image
	Mat hidden = hiddenImg.getPic();

	// get the dimension of the hidden image
	int hh = hiddenImg.getHeight();
	int hw = hiddenImg.getWidth();
	int hc = hiddenImg.getColors();

	for (int k = 0; k < c; k++) {
		for (int i = 0; i < hh; i++) {
			for (int j = 0; j < hw; j++) {
				if (picture_m[h * w * k + w * i + 4 * j] < 0)
					arr_hide[hh * hw * 4 * k + w * i + 4 * j] = (int)(-picture_m[h * w * k + w * i + 4 * j]) & 0x03;
				else
					arr_hide[hh * hw * 4 * k + w * i + 4 * j] = (int)(picture_m[h * w * k + w * i + 4 * j]) & 0x03;

				if (picture_m[h * w * k + w * i + 4 * j + 1] < 0)
					arr_hide[hh * hw * 4 * k + w * i + 4 * j + 1] = (int)(-picture_m[h * w * k + w * i + 4 * j + 1]) & 0x03;
				else
					arr_hide[hh * hw * 4 * k + w * i + 4 * j + 1] = (int)(picture_m[h * w * k + w * i + 4 * j + 1]) & 0x03;

				if (picture_m[h * w * k + w * i + 4 * j + 2] < 0)
					arr_hide[hh * hw * 4 * k + w * i + 4 * j + 2] = (int)(-picture_m[h * w * k + w * i + 4 * j + 2]) & 0x03;
				else
					arr_hide[hh * hw * 4 * k + w * i + 4 * j + 2] = (int)(picture_m[h * w * k + w * i + 4 * j + 2]) & 0x03;

				if (picture_m[h * w * k + w * i + 4 * j + 3] < 0)
					arr_hide[hh * hw * 4 * k + w * i + 4 * j + 3] = (int)(-picture_m[h * w * k + w * i + 4 * j + 3]) & 0x03;
				else
					arr_hide[hh * hw * 4 * k + w * i + 4 * j + 3] = (int)(picture_m[h * w * k + w * i + 4 * j + 3]) & 0x03;
			}
		}
	}

	for (int k = 0; k < c; k++) {
		for (int i = 0; i < hh; i++) {
			for (int j = 0; j < hw; j++) {
				hidden.at<cv::Vec3b>(i, j)[k] = (arr_hide[hh * hw * 4 * k + w * i + 4 * j] << 6) +
					(arr_hide[hh * hw * 4 * k + w * i + 4 * j + 1] << 4) +
					(arr_hide[hh * hw * 4 * k + w * i + 4 * j + 2] << 2) +
					(arr_hide[hh * hw * 4 * k + w * i + 4 * j + 3]);
			}
		}
	}

	namedWindow("Extracted Image", WINDOW_NORMAL);
	cv::imshow("Extracted Image", hidden);
	cv::waitKey();

}