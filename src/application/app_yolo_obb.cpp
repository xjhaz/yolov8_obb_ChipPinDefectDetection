
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
int yolo_obb_convert()
{
    INFO("Start convert process......\n");
    TRT::Mode mode = TRT::Mode::FP32;
    const string model = "best";
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
    cv::VideoCapture cap("/home/nvidia/tensorRT_Pro-YOLOv8/workspace/Video_00001.mp4");
    // std::string gstPipeline = "v4l2src device=/dev/video0 ! video/x-raw, format=BGRx ! videoconvert ! videoscale ! appsink";

    // cv::VideoCapture cap(gstPipeline, cv::CAP_GSTREAMER);

    // cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open video file." << std::endl;
        return -1;
    }
    // cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    auto engine = YoloOBB::create_infer(
        "/home/nvidia/tensorRT_Pro-YOLOv8/workspace/three.FP32.trtmodel", // engine file
        0,                                                                // gpu id
        0.75f,                                                            // confidence threshold
        0.3f,                                                             // nms threshold
        YoloOBB::NMSMethod::FastGPU,                                      // NMS method, fast GPU / CPU
        1024,                                                             // max objects
        false                                                             // preprocess use multi stream
    );
    if (engine == nullptr)
    {
        INFOE("Engine is nullptr");
        exit(-1);
    }
    cv::Mat image;
    while (true)
    {
        // auto start = std::chrono::high_resolution_clock::now();
        cap >> image;
        if (image.empty())
        {
            break; // End of video
        }
        if (image.empty())
        {
            INFOE("Image is empty");
            break;
        }
        // auto start1 = std::chrono::high_resolution_clock::now();
        auto boxes = engine->commit(image).get();
        // auto end1 = std::chrono::high_resolution_clock::now();

        // Calculate the duration
        // std::chrono::duration<double> duration1 = end1 - start1;

        // Print the duration in seconds
        // std::cout << "Time taken by function: " << duration1.count() << " seconds" << std::endl;

        for (auto &obj : boxes)
        {
            uint8_t b, g, r;
            tie(b, g, r) = iLogger::random_color(obj.class_label);

            // std::cout << dotalabels[obj.class_label] << std::endl;
            if (obj.class_label == 1)
            {
                auto corners = xywhr2xyxyxyxy(obj);
                cv::polylines(image, vector<vector<cv::Point>>{corners}, true, cv::Scalar(b, g, r), 2, 16);
                auto name = dotalabels[obj.class_label];
                auto caption = iLogger::format("%s %.2f", name, obj.confidence);
                int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
                cv::rectangle(image, cv::Point(corners[0].x, corners[0].y), cv::Point(corners[0].x + width, corners[0].y + 30), cv::Scalar(b, g, r), -1);
                cv::putText(image, caption, cv::Point(corners[0].x, corners[0].y + 30), 0, 1, cv::Scalar::all(0), 2, 16);
            }
        }
        // auto end = std::chrono::high_resolution_clock::now();

        // Calculate the duration
        // std::chrono::duration<double> duration = end - start;

        // Print the duration in seconds
        // std::cout << "Time taken by function: " << duration.count() << " seconds" << std::endl;

        cv::resize(image, image, cv::Size(1920 / 2, 1080 / 2));
        cv::imshow("Video", image);
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
