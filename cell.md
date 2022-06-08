```
cv::Mat MainWindow::cell()
{
    cv::Mat cell = cv::imread("D:/projects/image/1.tif", 0);
    imshow("cell",cell);

    //-----使用sobel算子初步提取细胞轮廓
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y, cell_sobal;
    Sobel(cell, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
//    imshow("x_sobal", abs_grad_x);

    Sobel(cell, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);
//    imshow("y_sobal", abs_grad_y);


    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, cell_sobal);
    imshow("all_sobal", cell_sobal);
    imwrite("D:/projects/cell_adapt.tif", cell_sobal);

    //使用alpha和beta对图片进行对比度和亮度调整
    Mat cell_bright;
    double alpha = 3.9;
    double beta = -96.0;

    // 直接使用循环遍历每一个像素，应用公式
//    double t1 = (double)getTickCount();
    for (int row=0; row<cell_sobal.rows; ++row)
        for(int col=0; col<cell_sobal.cols; ++col)
                cell_sobal.at<uchar>(row, col) = saturate_cast<uchar>(alpha * cell_sobal.at<uchar>(row, col) + beta);
//    double time1 = ((double)getTickCount() - t1) / getTickFrequency();

    // 调用 convertTo() 函数调整对比度和亮度
//    double t2 = (double)getTickCount();
    cell_sobal.convertTo(cell_bright, -1, alpha, beta);
//    double time2 = ((double)getTickCount() - t2) / getTickFrequency();
    imshow("2", cell_bright);


    //膨胀操作 膨胀两个像素，加粗白色部分
    double size = 2;
    cv::Mat element = getStructuringElement(MORPH_RECT, Size(size, size));
    cv::Mat cell_dilate;
    cell_dilate.create(cell_bright.size(), cell_bright.type());
    dilate(cell_bright, cell_dilate, element);


//    填充操作 使用floodfile函数，填补空缺部分
    // 从（0，0）点开始填充
    Mat cell_floodfill = cell_dilate.clone();
    floodFill(cell_floodfill, cv::Point(0, 0), Scalar(255));

    Mat im_floodfill_inv;
    bitwise_not(cell_floodfill, im_floodfill_inv);
    //合并图像
    Mat cell_im_out = (cell_dilate | im_floodfill_inv);

//    imshow("fill", cell_im_out);

    //删除细胞和边缘联通的区域
    //在黑白图像外面套一圈厚度为10的白边
    cv::Mat cell_imclearborder, tmp1;
    copyMakeBorder(cell_im_out, tmp1, 10, 10, 10, 10, BORDER_CONSTANT, Scalar(255));
    //然后从白边上的任意点用黑色填充白色
    cv::Rect ccomp;
    cv::Mat dst = cv::Mat::zeros(tmp1.rows +2, tmp1.cols +2, CV_8UC1);
    floodFill(tmp1, dst, Point(0, 0), Scalar(0), &ccomp, Scalar(0), Scalar(20));
    imshow("tmp1", tmp1);

    //裁剪图片，去掉之前添加的白边
    int c_x = tmp1.cols;
    int c_y = tmp1.rows;
    cv::Rect rect(10, 10, c_x - 20, c_y - 20);
    cell_imclearborder = tmp1(rect);
    imshow("tmp2", cell_imclearborder);



    //平滑处理图像
    cv::Mat element1 = getStructuringElement(MORPH_RECT,Size(2,2));
    Mat cell_erode1, cell_erode;    //定义接收腐蚀后图像的数据内存
    erode(cell_imclearborder,cell_erode1,element1);     //和matlab中的imerode相对应,腐蚀函数的调用过程
    erode(cell_erode1,cell_erode,element1);
    imshow("erode",cell_erode);
    imwrite("erode.tif", cell_erode);

    cv::Mat cellRGB;
    cvtColor(cell, cellRGB, COLOR_GRAY2RGB);


    //在原始图像上显示掩膜（下面两种方法2选1即可）

    //----以改变颜色的方式显示掩膜
//    for(int i=0; i<cell_erode.rows; i++)
//    {
//        for(int j=0; j<cell_erode.cols; j++)
//        {
//            if(cell_erode.at<uchar>(i, j) == 255){
//                if(cellRGB.at<Vec3b>(i-1, j-1)[2] < 205)
//                    cellRGB.at<Vec3b>(i-1, j-1)[2] += 50;
//                else
//                    cellRGB.at<Vec3b>(i-1, j-1)[2] = 255;
//            }
//        }
//    }
//    imwrite("cell_rgb.tif", cellRGB);

    //----以划线的方式显示掩膜
    for(int i=0; i<cell_erode.rows; i++)
    {
        for(int j=0; j<cell_erode.cols; j++)
        {
            if(cell_erode.at<uchar>(i, j) == 255 && cell_erode.at<uchar>(i+1,j) == 0)
                cellRGB.at<Vec3b>(i-1, j-1) = (255,255,255);

            if(cell_erode.at<uchar>(i, j) == 255 && cell_erode.at<uchar>(i+1,j+1) == 0)
                cellRGB.at<Vec3b>(i-1, j-1) = (255,255,255);

            if(cell_erode.at<uchar>(i, j) == 255 && cell_erode.at<uchar>(i,j) == 0)
                cellRGB.at<Vec3b>(i-1, j-1) = (255,255,255);

            if(cell_erode.at<uchar>(i, j) == 255 && cell_erode.at<uchar>(i,j+1) == 0)
                cellRGB.at<Vec3b>(i-1, j-1) = (255,255,255);

            if(cell_erode.at<uchar>(i, j) == 255 && cell_erode.at<uchar>(i-1,j) == 0)
                cellRGB.at<Vec3b>(i-1, j-1) = (255,255,255);

            if(cell_erode.at<uchar>(i, j) == 255 && cell_erode.at<uchar>(i-1,j-1) == 0)
                cellRGB.at<Vec3b>(i-1, j-1) = (255,255,255);

            if(cell_erode.at<uchar>(i, j) == 255 && cell_erode.at<uchar>(i,j-1) == 0)
                cellRGB.at<Vec3b>(i-1, j-1) = (255,255,255);
        }
    }

    imwrite("cell_rgb.tif", cellRGB);
    return cellRGB;
}
```