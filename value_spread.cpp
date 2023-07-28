#include <opencv2/opencv.hpp>
#include <Halide.h>
#include <chrono>
#include "build/value_spread_lib.h"

cv::Mat computeSpread(const cv::Mat& input)
{
    cv::Mat dilated, eroded;
    //every pixel of dilated will have the maximum of its 5x5 neighbors
    cv::dilate(input, dilated, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
    //every pixel of eroded will have the minimum of its 5x5 neighbors
    cv::erode(input, eroded, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
    return cv::abs(eroded - dilated);
}

cv::Mat computeSpreadHalideSlow(const cv::Mat& input_)
{
    using namespace Halide;
    // Declare some Vars to use below.
    Var x("x"), y("y");

    // Load a grayscale image to use as an input.
    Halide::Buffer<uint8_t> input = Halide::Runtime::Buffer<uint8_t>(
            input_.data, input_.cols, input_.rows);

    Func spread("spread"), clamped("clamped");
    Expr x_clamped = clamp(x, 0, input.width() - 1);
    Expr y_clamped = clamp(y, 0, input.height() - 1);
    clamped(x, y) = input(x_clamped, y_clamped);

    RDom box(-2, 5, -2, 5);
    // Compute the local maximum minus the local minimum:
    spread(x, y) = (maximum(clamped(x + box.x, y + box.y)) -
                    minimum(clamped(x + box.x, y + box.y)));

    // Compute the result in strips of 32 scanlines
    Var yo("yo"), yi("yi");
    spread.split(y, yo, yi, 32).parallel(yo);

    // Vectorize across x within the strips. This implicitly
    // vectorizes stuff that is computed within the loop over x in
    // spread, which includes our minimum and maximum helpers, so
    // they get vectorized too.
    spread.vectorize(x, 16);

    // We'll apply the boundary condition by padding each scanline
    // as we need it in a circular buffer (see lesson 08).
    clamped.store_at(spread, yo).compute_at(spread, yi);
    spread.compile_to_lowered_stmt("gradient.html", {}, HTML);
    Buffer<uint8_t> halide_result = spread.realize({input.width(), input.height()});
    //TODO realize to an existing buffer;
    //void Halide::Func::realize(Pipeline::RealizationArg outputs,
    //    const Target &target = Target(),
    //    const ParamMap &param_map = ParamMap::empty_map()
    //    ) 	
    return cv::Mat_<uint8_t>(halide_result.height(), halide_result.width(), 
             (uint8_t*)halide_result.get()->data()).clone();
}

cv::Mat computeSpreadHalideFast(const cv::Mat& input_)
{
    cv::Mat_<uint8_t> output_(input_.size(), input_.type());
    Halide::Buffer<uint8_t> input = Halide::Runtime::Buffer<uint8_t>(
            input_.data, input_.cols, input_.rows);
    Halide::Buffer<uint8_t> output = Halide::Runtime::Buffer<uint8_t>(
            output_.data, output_.cols, output_.rows);
    value_spread_lib(input.raw_buffer(), output.raw_buffer());
    return output_;
}

cv::Mat computeSpreadCppVersion(const cv::Mat& input)
{
    cv::Size size = input.size();
    cv::Mat c_result(size.height, size.width, CV_8UC1);
#ifndef __SSE2__
#error "you must have SSE2 for this code to work"
#endif

#pragma omp parallel for
    for (int yo = 0; yo < (size.height + 31) / 32; yo++) {
        int y_base = std::min(yo * 32, size.height - 32);

        // Compute clamped in a circular buffer of size 8
        // (smallest power of two greater than 5). Each thread
        // needs its own allocation, so it must occur here.

        int clamped_width = size.width + 4;
        uint8_t *clamped_storage = (uint8_t *)malloc(clamped_width * 8);

        for (int yi = 0; yi < 32; yi++) {
            int y = y_base + yi;

            uint8_t *output_row = c_result.ptr<uchar>(y);

            // Compute clamped for this scanline, skipping rows
            // already computed within this slice.
            int min_y_clamped = (yi == 0) ? (y - 2) : (y + 2);
            int max_y_clamped = (y + 2);
            for (int cy = min_y_clamped; cy <= max_y_clamped; cy++) {
                // Figure out which row of the circular buffer
                // we're filling in using bitmasking:
                uint8_t *clamped_row =
                    clamped_storage + (cy & 7) * clamped_width;

                // Figure out which row of the input we're reading
                // from by clamping the y coordinate:
                int clamped_y = std::min(std::max(cy, 0), size.height - 1);
                const uint8_t *input_row = input.ptr<const uint8_t>(clamped_y);

                // Fill it in with the padding.
                for (int x = -2; x < size.width + 2; x++) {
                    int clamped_x = std::min(std::max(x, 0), size.width - 1);
                    *clamped_row++ = input_row[clamped_x];
                }
            }

            // Now iterate over vectors of x for the pure step of the output.
            for (int x_vec = 0; x_vec < (size.width + 15) / 16; x_vec++) {
                int x_base = std::min(x_vec * 16, size.width - 16);

                // Allocate storage for the minimum and maximum
                // helpers. One vector is enough.
                __m128i minimum_storage, maximum_storage;

                // The pure step for the maximum is a vector of zeros
                maximum_storage = _mm_setzero_si128();

                // The update step for maximum
                for (int max_y = y - 2; max_y <= y + 2; max_y++) {
                    uint8_t *clamped_row =
                        clamped_storage + (max_y & 7) * clamped_width;
                    for (int max_x = x_base - 2; max_x <= x_base + 2; max_x++) {
                        __m128i v = _mm_loadu_si128(
                            (__m128i const *)(clamped_row + max_x + 2));
                        maximum_storage = _mm_max_epu8(maximum_storage, v);
                    }
                }

                // The pure step for the minimum is a vector of
                // ones. Create it by comparing something to
                // itself.
                minimum_storage = _mm_cmpeq_epi32(_mm_setzero_si128(),
                                                  _mm_setzero_si128());

                // The update step for minimum.
                for (int min_y = y - 2; min_y <= y + 2; min_y++) {
                    uint8_t *clamped_row =
                        clamped_storage + (min_y & 7) * clamped_width;
                    for (int min_x = x_base - 2; min_x <= x_base + 2; min_x++) {
                        __m128i v = _mm_loadu_si128(
                            (__m128i const *)(clamped_row + min_x + 2));
                        minimum_storage = _mm_min_epu8(minimum_storage, v);
                    }
                }

                // Now compute the spread.
                __m128i spread = _mm_sub_epi8(maximum_storage, minimum_storage);

                // Store it.
                _mm_storeu_si128((__m128i *)(output_row + x_base), spread);
            }
        }
        free(clamped_storage);
    }
    return c_result;
}

//using namespace Halide;
//class ComputeSpreadHalide
//{
//    Func spread;
//ComputeSpreadHalide()
//{
//    Var x("x"), y("y");
//}
//cv::Mat operator()(const cv::Mat& input_)
//{
//    // Declare some Vars to use below.
//
//    // Load a grayscale image to use as an input.
//    Halide::Buffer<uint8_t> input = Halide::Runtime::Buffer<uint8_t>(
//            input_.data, input_.cols, input_.rows);
//
//    Func clamped;
//    Expr x_clamped = clamp(x, 0, input.width() - 1);
//    Expr y_clamped = clamp(y, 0, input.height() - 1);
//    clamped(x, y) = input(x_clamped, y_clamped);
//
//    RDom box(-2, 5, -2, 5);
//    // Compute the local maximum minus the local minimum:
//    spread(x, y) = (maximum(clamped(x + box.x, y + box.y)) -
//                    minimum(clamped(x + box.x, y + box.y)));
//
//    // Compute the result in strips of 32 scanlines
//    Var yo, yi;
//    spread.split(y, yo, yi, 32).parallel(yo);
//
//    // Vectorize across x within the strips. This implicitly
//    // vectorizes stuff that is computed within the loop over x in
//    // spread, which includes our minimum and maximum helpers, so
//    // they get vectorized too.
//    spread.vectorize(x, 16);
//
//    // We'll apply the boundary condition by padding each scanline
//    // as we need it in a circular buffer (see lesson 08).
//    clamped.store_at(spread, yo).compute_at(spread, yi);
//
//    Buffer<uint8_t> halide_result = spread.realize({input.width(), input.height()});
//    //TODO realize to an existing buffer;
//    //void Halide::Func::realize(Pipeline::RealizationArg outputs,
//    //    const Target &target = Target(),
//    //    const ParamMap &param_map = ParamMap::empty_map()
//    //    ) 	
//    return cv::Mat_<uint8_t>(halide_result.height(), halide_result.width(), 
//             (uint8_t*)halide_result.get()->data()).clone();
//}
//}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::cerr << "not enough arguments, we need an input image name" << std::endl;
        return 1;
    }
    cv::Mat input = cv::imread(argv[1], cv::IMREAD_ANYDEPTH);
    if(input.channels() == 3)
        cv::cvtColor(input, input, cv::COLOR_BGR2GRAY);

    if(input.depth() != CV_8U)
        input.convertTo(input, CV_8U);
    std::cout << "input.channels() " <<input.channels() << std::endl; 

    auto run_and_measure = [&](auto function, auto name)
    {
        auto t_start = std::chrono::high_resolution_clock::now();
        cv::Mat spread = function(input);
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout << name << " took " << elapsed_time_ms << " milliseconds" << std::endl;
        cv::imwrite(std::string(name) + ".jpg", spread);
    };
    run_and_measure(computeSpread, "spread");
    run_and_measure(computeSpreadCppVersion, "spread_cpp");
    run_and_measure(computeSpreadHalideSlow, "spread_by_halide_slow");
    run_and_measure(computeSpreadHalideFast, "spread_by_halide_fast");
    return 0;
}

