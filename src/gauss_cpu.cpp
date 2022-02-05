#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core.hpp>
#include <iostream>
#include <string>
#include <chrono>
using namespace cv;
using namespace std;
//https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1

int main(int argc, char **argv)
{
string input_file="images/bike_6000x4000.jpg";
string output_file="images/iesire_bike_cpu.jpg";

if (argc == 3) {
    input_file  = std::string(argv[1]);
    output_file = std::string(argv[2]);
 }
else {
    cout << "Input file or output file not specified. Using default ones" << endl;
}
Mat image = cv::imread(input_file.c_str());
  if (image.empty()) {
    std::cerr << "Couldn't open file: " << input_file << std::endl;
    exit(1);
  }

Mat output_image = image.clone();
auto start = std::chrono::high_resolution_clock::now();
GaussianBlur(image, output_image, Size( 5, 5), 4.0, 0 );//applying Gaussian filter
auto stop = std::chrono::high_resolution_clock::now();

std::cout << "Done in " << std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count() << " ms" << std::endl;
imwrite(output_file.c_str(), output_image);

return 0;
} 
