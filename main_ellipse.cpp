#include <Halide.h>
#include <opencv2/opencv.hpp>
int main()
{
    using namespace Halide;
    Func gradient;
    Var x, y, c;
    Expr x_sq = cast<float>(x) * x, y_sq = cast<float>(y) * y;
    float a_sq = 200.f * 200.f, b_sq = 150.f * 150.f;
    gradient(c, x, y) = cast<uint8_t>(cast<int32_t>(
                (x_sq / a_sq + y_sq / b_sq) * 255.f + c * 85.f) & 0xff);
    Halide::Buffer<> output = gradient.realize({3, 320, 240});
    cv::Mat_<cv::Vec3b> wrapper(240, 320, (cv::Vec3b*)output.get()->data());
    cv::imwrite("gradient.jpg", wrapper);
}
