//
//  main.cpp
//  GausLapPyr
//
//  Created on 2/5/18.
//  Copyright © 2018. All rights reserved.
//

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

void convolve(Mat src, float kernelX[], Mat& dst, int sizeKernel){
    
    //calculate padding needed to keep src image the same
    double pad = (sizeKernel - 1.0)/2.0;
    
    //add padding
    Mat paddedImg = Mat(src.rows+2*pad, src.cols+2*pad, src.type(), double(0.0));
    copyMakeBorder(src, paddedImg, pad, pad, pad, pad, BORDER_REFLECT);
    
    //create temp matrices for convolution
    Mat convolvedImg = Mat(paddedImg.rows, paddedImg.cols, paddedImg.type(), double(0.0));
    Mat convolvedImgX = Mat(paddedImg.rows, paddedImg.cols, paddedImg.type(), double(0.0));
    
    double sum;
    
    //convolve with horizontal kernel over the rows of the images
    for(int r=0; r<paddedImg.rows; r++){
        
        for( int c=0; c<paddedImg.cols-2*pad; c++){
            sum = 0.0;
            for(int k=0; k<sizeKernel; k++){
                sum += paddedImg.at<uchar>(r,c+k) * kernelX[k];
            }
            //store convolution value into temp matrix
            convolvedImgX.at<uchar>(r,c+pad) = sum;
        }
    }
    
    //convolve on previous result columns, using vertical kernel
    for(int c=pad; c<paddedImg.cols-pad; c++){
        for( int r=0; r<paddedImg.rows-2*pad; r++){
            sum = 0.0;
            for(int k=0; k<sizeKernel; k++){
                sum += convolvedImgX.at<uchar>(r+k,c) * kernelX[k];
            }
            convolvedImg.at<uchar>(r+pad,c) = sum;
        }
    }
    
    //remove added padding
    convolvedImg(Range(0+pad, convolvedImg.rows - pad), Range(0+pad, convolvedImg.cols-pad)).copyTo(dst);
}

void applyGaussianFilter(Mat src, Mat& dst){
    
        //create kernel coefficients (coeff used in build-in fct pyrDown of opencv)
        float kernelX[5] = {1.0/16.0,4.0/16.0,6.0/16.0,4.0/16.0,1.0/16.0};
        int sizeKernel = 5;
    
        //convolve img src with kernel
        convolve(src, kernelX, dst, sizeKernel);
    
    /*
        double data[25] = {1.0,4.0,6.0,4.0,1.0,4.0,16.0,24.0,16.0,4.0,6.0,24.0,36.0,24.0,6.0,4.0,16.0,24.0,16.0,4.0,1.0,4.0,6.0,4.0,1.0};
        double coeff = 1.0/256.0;
        
        //apply coeff to Mat
        Mat kernel = Mat(5, 5, CV_64F, data);
        kernel *= coeff;
        
        //convolve img scr with kernel
        filter2D(src, dst, -1, kernel, Point(-1,-1), 0, BORDER_DEFAULT);
    */
}

void downSampleImg(Mat src, Mat& dst){
   
    //remove every even cols
    Mat removeEvenCols;
    for (int i=1; i<src.cols; i+=2){ removeEvenCols.push_back(src.col(i));}
    
    //reshape and transpose
    removeEvenCols = removeEvenCols.reshape(0,src.cols/2);
    removeEvenCols = removeEvenCols.t();
    
    //remove every even rows
    Mat removeEvenRows;
    for(int i=1; i<removeEvenCols.rows; i+=2){ removeEvenRows.push_back(removeEvenCols.row(i));}
    
    dst = removeEvenRows;
}

void upSampleImg(Mat src, Mat& dst, Size nextLevelSize){
 
    Mat newSizeCols;
    Mat newSizeRows;
    
    //Make sure the size of the image will match with next level pyramid
    if(src.size()*2 != nextLevelSize){
        //create new columns sized Matrix filled with zeros
        newSizeCols = Mat(src.rows, src.cols*2+1, src.type(), double(0.0));
        
        //fill odd cols in new Mat by src cols
        for (int i=0; i<src.cols; i+=1){
            src.col(i).copyTo(newSizeCols.col(i*2 +1));
        }
        
        //create new rows sized Matrix filled with zeros
        newSizeRows = Mat(newSizeCols.rows*2+1, newSizeCols.cols, src.type(), double(0.0));
        
        //fill odd cols in new Mat by src cols
        for (int i=0; i<newSizeCols.rows; i+=1){
            newSizeCols.row(i).copyTo(newSizeRows.row(i*2 +1));
        }
    }
    else{
        //create new columns sized Matrix filled with zeros
        newSizeCols = Mat(src.rows, src.cols*2, src.type(), double(0.0));
        
        //fill odd cols in new Mat by src cols
        for (int i=0; i<src.cols; i+=1){
            src.col(i).copyTo(newSizeCols.col(i*2 +1));
        }
        
        //create new rows sized Matrix filled with zeros
        newSizeRows = Mat(newSizeCols.rows*2, newSizeCols.cols, src.type(), double(0.0));
        
        //fill odd cols in new Mat by src cols
        for (int i=0; i<newSizeCols.rows; i+=1){
            newSizeCols.row(i).copyTo(newSizeRows.row(i*2 +1));
        }
    }
    
    dst = newSizeRows;
}

int main(int argc, const char * argv[]) {

    Mat imgInput = imread("lena.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    
    //check if image was correctly open
    if(imgInput.empty()){
        printf("Error opening image.\n");
        return -1;
    }
    
    //resize img to have even rows and cols
    if(imgInput.rows % 2 != 0){
        resize(imgInput, imgInput, Size(imgInput.cols, imgInput.rows-1));
    }
    if(imgInput.cols % 2 != 0){
        resize(imgInput, imgInput, Size(imgInput.cols-1, imgInput.rows));
    }
    
    //imwrite("lena_gray.jpg", imgInput);
    
    //imshow("imgSrc", imgInput);
    
    int levelPyr;
    Mat src = imgInput;
    Mat smooth;
    Mat gaussian;
    
    Mat gaussianPyr[4];
    
    //Creation of Gaussian pyramids
    for(levelPyr=0; levelPyr<4; levelPyr++){
        
        //blur image
        applyGaussianFilter(src, smooth);
        
        //divide image size by 2
        downSampleImg(smooth, gaussian);
        imshow("gaussian_"+to_string(levelPyr+1), gaussian);
        //imwrite("lena_gaussian_"+to_string(levelPyr+1)+".jpg", gaussian);
        
        //store the image
        gaussianPyr[levelPyr] = gaussian;
        src = gaussian;
    }
    
    Mat upimg;
    Mat laplacianTemp;
    Mat laplacian;
    Mat laplacianPyr[4];
    
    //Creation of Laplacian Pyramid
    for(levelPyr=0; levelPyr<4; levelPyr++){
        
        //use ImgInput to get difference
        if(levelPyr == 0){
            upSampleImg(gaussianPyr[levelPyr], upimg, imgInput.size());
            assert(upimg.size() == imgInput.size());
            applyGaussianFilter(upimg, laplacianTemp);
            laplacianTemp *= 4;
            laplacian = imgInput - laplacianTemp;
        }
        else {
            //multiple size of image by 2
            upSampleImg(gaussianPyr[levelPyr], upimg, gaussianPyr[levelPyr-1].size());
            assert(upimg.size() == gaussianPyr[levelPyr-1].size());
            
            //extrapolate empty pixel using blurring
            applyGaussianFilter(upimg, laplacianTemp);
            
            //multiply pixel value by 4 to lighten the image
            laplacianTemp *= 4;
        
            //calculate difference of gaussian = laplacian
            laplacian = gaussianPyr[levelPyr-1] - laplacianTemp;
        }
    
        //enhanced contrast to see high-pass pixels
        Mat laplacianEnhanced;
        laplacian.convertTo(laplacianEnhanced, -1, 8, 0);
        imshow("laplacian_"+to_string(levelPyr), laplacianEnhanced);
        //imwrite("lena_laplacian_"+to_string(levelPyr)+".jpg", laplacianEnhanced);
    
        laplacianPyr[levelPyr]=laplacianEnhanced;
        src = laplacianEnhanced;
    }
    
    
    // Wait until user press some key
    waitKey(0);
    return 0;
}
