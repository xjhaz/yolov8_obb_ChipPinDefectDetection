
#include <stdio.h>
#include <string.h>
#include <common/ilogger.hpp>
#include <functional>

int yolo_obb_convert();
int yolo_obb_infer();

int main(int argc, char **argv)
{
    if (argc > 1)
    { // 确保有至少一个参数被传入
        if (strcmp(argv[1], "infer") == 0)
        {
            yolo_obb_infer();
        }
        else if (strcmp(argv[1], "convert") == 0)
        {
            yolo_obb_convert();
        }
        else
        {
            INFOE("Invalid argument. Use 'infer' or 'convert'.\n");
        }
    }
    else
    {
        INFO("No arguments provided. Defult to use infer.\n");
        yolo_obb_infer();
    }
    return 0;
}
