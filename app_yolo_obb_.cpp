
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "app_yolo_obb/yolo_obb.hpp"
#include "tools/zmq_remote_show.hpp"
#include <chrono>
using namespace std;

static const char *dotalabels[] = {
    "1", "broken"};

bool
    requires(const char *name);

static void append_to_file(const string &file, const string &data)
{
    FILE *f = fopen(file.c_str(), "a+");
    if (f == nullptr)
    {
        INFOE("Open %s failed.", file.c_str());
        return;
    }

    fprintf(f, "%s\n", data.c_str());
    fclose(f);
}

static vector<cv::Point> xywhr2xyxyxyxy(const YoloOBB::Box &box)
{
    float cos_value = std::cos(box.angle);
    float sin_value = std::sin(box.angle);

    float w_2 = box.width / 2, h_2 = box.height / 2;
    float vec1_x = w_2 * cos_value, vec1_y = w_2 * sin_value;
    float vec2_x = -h_2 * sin_value, vec2_y = h_2 * cos_value;

    vector<cv::Point> corners;
    corners.push_back(cv::Point(box.center_x + vec1_x + vec2_x, box.center_y + vec1_y + vec2_y));
    corners.push_back(cv::Point(box.center_x + vec1_x - vec2_x, box.center_y + vec1_y - vec2_y));
    corners.push_back(cv::Point(box.center_x - vec1_x - vec2_x, box.center_y - vec1_y - vec2_y));
    corners.push_back(cv::Point(box.center_x - vec1_x + vec2_x, box.center_y - vec1_y + vec2_y));

    return corners;
}

static void test(TRT::Mode mode, const string &model)
{
    int deviceid = 0;
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(deviceid);

    auto int8process = [=](int current, int count, const vector<string> &files, shared_ptr<TRT::Tensor> &tensor)
    {
        INFO("Int8 %d / %d", current, count);

        for (int i = 0; i < files.size(); ++i)
        {
            auto image = cv::imread(files[i]);
            YoloOBB::image_to_tensor(image, tensor, i);
        }
    };

    const char *name = model.c_str();
    INFO("===================== test YoloV8-OBB %s %s ==================================", mode_name, name);

    if (not requires(name))
        return;

    string onnx_file = iLogger::format("%s.onnx", name);
    string model_file = iLogger::format("%s.%s.trtmodel", name, mode_name);
    int test_batch_size = 16;

    if (not iLogger::exists(model_file))
    {
        TRT::compile(
            mode,            // FP32、FP16、INT8
            test_batch_size, // max batch size
            onnx_file,       // source
            model_file,      // save to
            {},
            int8process,
            "inference");
    }
}

int app_yolo_obb()
{
    test(TRT::Mode::FP32, "three");
    return 0;
}