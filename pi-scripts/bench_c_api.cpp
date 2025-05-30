// pi-scripts/benchmark_tflite_cpp.cpp
#include <cstdio>
#include <chrono>
#include <vector>
#include <sys/stat.h>
#include <numeric> // For std::iota (optional for dummy data)
#include <algorithm> // For std::fill (optional for dummy data)

// TensorFlow Lite C++ API headers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h" // For BuiltinOpResolver
#include "tensorflow/lite/model.h"           // For FlatBufferModel
#include "tensorflow/lite/optional_debug_tools.h" // For PrintInterpreterState


// Helper function to get tensor size
size_t GetTensorSize(const TfLiteIntArray* dims) {
    size_t size = 1;
    for (int i = 0; i < dims->size; ++i) {
        size *= dims->data[i];
    }
    return size;
}

int main(int argc, char** argv) {
    if (argc != 4 && argc != 3) { // 允许不提供 sample.npy，如果我们要用dummy data
        std::fprintf(stderr, "Usage: %s model.tflite [sample.npy] warmup_runs [num_benchmark_runs]\n", argv[0]);
        std::fprintf(stderr, "If sample.npy is omitted, dummy zero data will be used.\n");
        std::fprintf(stderr, "If num_benchmark_runs is omitted, it defaults to 100.\n");
        return 1;
    }
    const char* model_path = argv[1];
    // const char* sample_path = argv[2]; // 我们将使其可选
    int warmup_runs = std::atoi(argv[argc - 2]); // warmup_runs 现在是倒数第二个参数
    int num_benchmark_runs = (argc == 4) ? std::atoi(argv[argc - 1]) : 100; // benchmark_runs 是最后一个，或默认100


    std::printf("Debug: Model path: %s\n", model_path);
    std::printf("Debug: Warmup runs: %d\n", warmup_runs);
    std::printf("Debug: Benchmark runs: %d\n", num_benchmark_runs);

    // 1. 加载模型
    std::printf("Debug: Loading model...\n");
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);
    if (!model) {
        std::fprintf(stderr, "Error: Failed to load model %s\n", model_path);
        return 1;
    }
    std::printf("Debug: Model loaded successfully.\n");

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter); // Build the interpreter
    if (!interpreter) {
        std::fprintf(stderr, "Error: Failed to build interpreter.\n");
        return 1;
    }
    std::printf("Debug: Interpreter built.\n");

    // 设置线程数 (可选，默认为1，除非TFLite内部有不同默认或模型指定)
    // interpreter->SetNumThreads(1);

    std::printf("Debug: Allocating tensors...\n");
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::fprintf(stderr, "Error: Failed to allocate tensors.\n");
        return 1;
    }
    std::printf("Debug: Tensors allocated.\n");

    // 获取输入张量信息
    int input_idx = interpreter->inputs()[0];
    TfLiteTensor* input_tensor_ptr = interpreter->tensor(input_idx);
    TfLiteIntArray* input_dims = input_tensor_ptr->dims;

    std::printf("Debug: Input tensor (index %d):\n", input_idx);
    std::printf("  Name: %s\n", input_tensor_ptr->name ? input_tensor_ptr->name : "N/A");
    //  (需要一个函数来打印 TfLiteType，或者假设它是 float32)
    std::printf("  Type: %s\n", TfLiteTypeGetName(input_tensor_ptr->type)); // TfLiteTypeGetName is C API, ensure it's linkable or use a fallback
    std::printf("  Dimensions: ");
    for (int i = 0; i < input_dims->size; ++i) {
        std::printf("%d%s", input_dims->data[i], (i < input_dims->size - 1) ? "x" : "");
    }
    std::printf("\n");

    // 准备输入数据 (使用全零的 dummy data)
    // 假设输入类型是 float32，如果不是，需要根据 input_tensor_ptr->type 处理
    if (input_tensor_ptr->type != kTfLiteFloat32) {
        std::fprintf(stderr, "Error: This example currently only supports kTfLiteFloat32 input type.\n");
        return 1;
    }
    float* input_tensor_data = interpreter->typed_input_tensor<float>(0); // 获取第一个输入张量的指针
    size_t input_tensor_elements = GetTensorSize(input_dims);
    std::vector<float> dummy_input_data(input_tensor_elements, 0.0f); // 创建全零数据

    // 2. 读取 sample.npy 的逻辑被替换为使用 dummy data
    // 如果要从文件读取，你需要实现或使用库来解析npy文件并填充 dummy_input_data

    // 3. 预热
    std::printf("Debug: Warming up for %d runs...\n", warmup_runs);
    for (int i = 0; i < warmup_runs; ++i) {
        // 填充输入张量数据 - 对于每次 invoke，如果内容不变，只需填充一次
        // 但如果内容会变（如此处循环中），或者为了模拟真实场景，每次都填
        // 在这个dummy data的例子里，数据是固定的，可以在循环外填充一次
        // 或者如果每次 invoke 的数据不同，就在这里从 dummy_input_data 的不同部分取值
        // 这里我们假设每次预热和 benchmark 都用相同的 dummy_input_data
        for(size_t j=0; j<input_tensor_elements; ++j) {
            input_tensor_data[j] = dummy_input_data[j];
        }
        if (interpreter->Invoke() != kTfLiteOk) {
            std::fprintf(stderr, "Error: Failed to invoke interpreter during warmup.\n");
            return 1;
        }
    }
    std::printf("Debug: Warmup complete.\n");

    // 4. Benchmark
    std::printf("Debug: Starting benchmark for %d runs...\n", num_benchmark_runs);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_benchmark_runs; ++i) {
        // 同样，如果输入数据固定，可以不在循环内反复填充
        // 但如果模拟不同输入，这里应该填充新的数据
        for(size_t j=0; j<input_tensor_elements; ++j) {
            input_tensor_data[j] = dummy_input_data[j]; // 使用相同的dummy data
        }
        if (interpreter->Invoke() != kTfLiteOk) {
            std::fprintf(stderr, "Error: Failed to invoke interpreter during benchmark run %d.\n", i);
            return 1;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("\nBenchmark Results:\n");
    std::printf("Total time for %d runs: %.2f ms\n", num_benchmark_runs, total_ms);
    std::printf("Average inference time: %.3f ms\n", total_ms / num_benchmark_runs);

    // 5. 模型大小
    struct stat st;
    if (stat(model_path, &st) == 0) {
        double size_mb = st.st_size / 1024.0 / 1024.0;
        std::printf("Model size: %.2f MB\n", size_mb);
    } else {
        std::fprintf(stderr, "Warning: Could not get model file size.\n");
    }

    std::printf("Debug: Benchmark finished successfully.\n");
    return 0;
}