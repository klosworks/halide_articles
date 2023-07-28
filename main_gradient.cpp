#include <Halide.h>
#include <opencv2/opencv.hpp>
int main()
{
    Halide::Func gradient;
    Halide::Var x, y;
    gradient(x, y) = Halide::cast<uint8_t>((x + 2 * y) & 0xFF);
    Halide::Buffer<> output = gradient.realize({320, 240});
    cv::Mat_<uint8_t> wrapper(output.height(), output.width(), 
             (uint8_t*)output.get()->data());
    cv::imwrite("gradient.jpg", wrapper);
}
