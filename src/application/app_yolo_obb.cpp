
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "app_yolo_obb/yolo_obb.hpp"
#include "tools/zmq_remote_show.hpp"
#include <chrono>
#include <fstream>
#include <yaml-cpp/yaml.h>
using namespace std;

TRT::Mode getModeFromString(const std::string &modeStr)
{
    static std::map<std::string, TRT::Mode> modeMap = {
        {"FP32", TRT::Mode::FP32},
        {"FP16", TRT::Mode::FP16},
        {"INT8", TRT::Mode::INT8}};

    auto it = modeMap.find(modeStr);
    if (it != modeMap.end())
    {
        return it->second;
    }
    else
    {
        throw std::runtime_error("Unknown mode string: " + modeStr);
    }
}
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

int yolo_obb_convert()
{
    INFO("Start convert process......\n");

    std::ifstream file("../config/config_convert.yaml");
    YAML::Node config = YAML::Load(file);
    TRT::Mode mode = getModeFromString(config["mode"].as<std::string>());
    const string model = config["model"].as<std::string>();

    int deviceid = 0;
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(deviceid);

    auto int8process = [=](int current, int count, const vector<string> &files, shared_ptr<TRT::Tensor> &tensor)
    {
        INFO("Int8 %d / %d", current, count);

        for (int i = 0; i < files.size(); ++i)
        {
            auto image = cv::imread(files[i]);
            cv::resize(image, image, cv::Size(640, 640));
            YoloOBB::image_to_tensor(image, tensor, i);
        }
    };

    const char *name = model.c_str();
    INFO("===================== convert YoloV8-OBB %s %s ==================================", mode_name, name);

    if (not requires(name))
        return -1;

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
            "inference_obb",
            "");
    }
    else
    {
        INFO("Model already exists. If you want to re-convert the model, please delete or rename the old model.\n");
    }
    return 0;
}

int yolo_obb_infer()
{
    INFO("Start infer process......\n");
    std::ifstream file("../config/config_infer.yaml");
    YAML::Node config = YAML::Load(file);
    int source_mode = config["source_mode"].as<int>();
    cv::VideoCapture cap;
    if (source_mode == 0)
    {
        cap.open(config["video"].as<int>());
    }
    else if(source_mode == 1)
    {
        cap.open(config["video_path"].as<std::string>());
    }
    else 
    {
        INFOE("Unknown source mode detected. Please verify the 'mode' setting in your 'config_infer.yaml' file.");
        return -1;
    }
    std::string engine_file = config["engine_file"].as<std::string>();
    int gpu_id = config["gpu_id"].as<int>();
    float confidence_threshold = config["confidence_threshold"].as<float>();
    float nms_threshold = config["nms_threshold"].as<float>();
    YoloOBB::NMSMethod nms_method = config["nms_method"].as<std::string>() == "FastGPU" ? YoloOBB::NMSMethod::FastGPU : YoloOBB::NMSMethod::CPU; // 假设只有这两种 NMS 方法
    int max_objects = config["max_objects"].as<int>();
    bool preprocess_multi_stream = config["preprocess_multi_stream"].as<bool>();

    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open video file." << std::endl;
        return -1;
    }
    auto engine = YoloOBB::create_infer(
        engine_file,
        gpu_id,
        confidence_threshold,
        nms_threshold,
        nms_method,
        max_objects,
        preprocess_multi_stream);
    if (engine == nullptr)
    {
        INFOE("Engine is nullptr");
        exit(-1);
    }
    cv::Mat frame;
    while (true)
    {
        cap >> frame;
        if (frame.empty())
        {
            break; 
        }
        if (frame.empty())
        {
            INFOE("frame is empty");
            break;
        }
        auto boxes = engine->commit(frame).get();

        for (auto &obj : boxes)
        {
            uint8_t b, g, r;
            tie(b, g, r) = iLogger::random_color(obj.class_label);
            if (obj.class_label == 1)
            {
                auto corners = xywhr2xyxyxyxy(obj);
                cv::polylines(frame, vector<vector<cv::Point>>{corners}, true, cv::Scalar(b, g, r), 2, 16);
                auto name = dotalabels[obj.class_label];
                auto caption = iLogger::format("%s %.2f", name, obj.confidence);
                int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
                cv::rectangle(frame, cv::Point(corners[0].x, corners[0].y), cv::Point(corners[0].x + width, corners[0].y + 30), cv::Scalar(b, g, r), -1);
                cv::putText(frame, caption, cv::Point(corners[0].x, corners[0].y + 30), 0, 1, cv::Scalar::all(0), 2, 16);
            }
        }

        cv::resize(frame, frame, cv::Size(frame.cols / 2, frame.rows / 2));
        cv::imshow("Video", frame);
        if (cv::waitKey(1) >= 0)
        {
            break; // Stop if any key is pressed
        }
    }
    engine.reset();
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
