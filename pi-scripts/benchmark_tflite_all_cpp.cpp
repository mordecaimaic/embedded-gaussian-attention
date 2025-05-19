// benchmark_tflite_all_cpp_v2.cpp
// (Using the logic from our "v3" iteration, adapted for fixed-length TFLite input)
// Assumes TFLite model expects input like [1,1,40,98] (or a similar fixed time length)
// and NPY data is prepared accordingly, e.g., (N, 1, 40, 98).

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <sys/stat.h>
#include <sys/resource.h>
#include <stdexcept>
#include <cmath> // For std::round, std::max, std::min

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/evaluation/utils.h" // For TfLiteTypeGetName

// --- NPY Loading Function ---
std::vector<float> load_npy(const std::string &path, std::vector<int> &shape_out) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open NPY file: " + path);
    }
    char magic[6];
    file.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") {
        throw std::runtime_error("Not a valid NPY file (magic string mismatch): " + path);
    }
    char version[2];
    file.read(version, 2);
    if (version[0] != 1 || version[1] != 0) {
         std::cerr << "Warning: NPY file version is not 1.0. Parsing might fail. Version: "
                   << (int)version[0] << "." << (int)version[1] << std::endl;
    }
    uint16_t header_len_val;
    file.read(reinterpret_cast<char*>(&header_len_val), sizeof(header_len_val));
    std::string header(header_len_val, '\0');
    file.read(&header[0], header_len_val);

    shape_out.clear();
    size_t shape_start = header.find("'shape': (");
    if (shape_start == std::string::npos) {
        throw std::runtime_error("Could not find 'shape': ( in NPY header: " + header);
    }
    shape_start += std::string("'shape': (").length();
    size_t shape_end = header.find(")", shape_start);
    if (shape_end == std::string::npos) {
        throw std::runtime_error("Could not find closing ')' for shape in NPY header: " + header);
    }
    std::string shape_str = header.substr(shape_start, shape_end - shape_start);
    std::stringstream ss_shape(shape_str);
    std::string segment;
    while(std::getline(ss_shape, segment, ',')) {
        segment.erase(0, segment.find_first_not_of(" \t\n\r\f\v"));
        segment.erase(segment.find_last_not_of(" \t\n\r\f\v") + 1);
        if (!segment.empty()) {
            try {
                shape_out.push_back(std::stoi(segment));
            } catch (const std::invalid_argument& ia) {
                throw std::runtime_error("Invalid number in shape string: '" + segment + "' from '" + shape_str + "'");
            } catch (const std::out_of_range& oor) {
                throw std::runtime_error("Number out of range in shape string: '" + segment + "' from '" + shape_str + "'");
            }
        }
    }
    if (shape_out.empty()) {
        throw std::runtime_error("Parsed shape is empty from NPY header: " + header);
    }
    if (header.find("'descr': '<f4'") == std::string::npos && header.find("'descr': dtype('float32')") == std::string::npos ) {
        std::cerr << "Warning: NPY file descr is not '<f4>' or float32. Assuming float32 data. Header: " << header << std::endl;
    }
    size_t total_elements = 1;
    for (int dim : shape_out) {
        if (dim <= 0) throw std::runtime_error("Invalid dimension in shape: " + std::to_string(dim));
        total_elements *= dim;
    }
    std::vector<float> data(total_elements);
    file.read(reinterpret_cast<char*>(data.data()), total_elements * sizeof(float));
    if (file.fail()) {
        throw std::runtime_error("Failed to read data from NPY file: " + path);
    }
    std::cerr << "Debug: Loaded NPY '" << path << "' shape=[";
    for (size_t i = 0; i < shape_out.size(); ++i) {
        std::cerr << shape_out[i] << (i < shape_out.size() - 1 ? "," : "");
    }
    std::cerr << "] total_elements=" << total_elements << std::endl;
    return data;
}

// --- Helper to get tensor size ---
static size_t GetTensorSize(const TfLiteIntArray* dims) {
    if (!dims) return 0;
    size_t sz = 1;
    for (int i = 0; i < dims->size; ++i) {
        if (dims->data[i] <= 0) return 0;
        sz *= dims->data[i];
    }
    return sz;
}

// --- Stats structure ---
struct BenchmarkStats {
    double avg_latency_ms;
    double cpu_usage_percent;
    double ram_usage_mb;
    double model_size_mb;
};

// --- Core benchmarking function ---
BenchmarkStats measure_model_performance(
        const std::string& model_path,
        const std::vector<float>& full_npy_data,
        const std::vector<int>& full_npy_shape,
        int warmup_iterations) {

    using namespace tflite;

    // --- Validations ---
    if (full_npy_shape.empty() || full_npy_shape.size() != 4) {
        throw std::runtime_error("NPY data must be 4-dimensional (N,C,H,W).");
    }
    int num_total_npy_samples = full_npy_shape[0];
    if (num_total_npy_samples <= 0) {
        throw std::runtime_error("NPY data has no samples (N=0).");
    }
    int npy_C = full_npy_shape[1];
    int npy_H = full_npy_shape[2];
    int npy_W = full_npy_shape[3];
    size_t elements_per_npy_sample = static_cast<size_t>(npy_C) * npy_H * npy_W;

    std::cerr << "\nDebug: Evaluating model '" << model_path << "'" << std::endl;
    std::cerr << "Debug: NPY samples to process: " << num_total_npy_samples
              << ", Warmup iterations: " << warmup_iterations << std::endl;
    std::cerr << "Debug: NPY sample shape (C,H,W after N): (" << npy_C << "," << npy_H << "," << npy_W
              << "), elements per NPY sample: " << elements_per_npy_sample << std::endl;

    // --- 1. Load TFLite Model ---
    std::unique_ptr<FlatBufferModel> model = FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) throw std::runtime_error("Failed to load TFLite model: " + model_path);
    std::cerr << "Debug: Model loaded successfully." << std::endl;

    // --- 2. Build Interpreter ---
    ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<Interpreter> interpreter;
    InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) throw std::runtime_error("Failed to build TFLite interpreter.");
    std::cerr << "Debug: Interpreter created." << std::endl;

    // Optional: Set Num Threads (Uncomment and adjust if needed)
    // int num_threads = 2; // Example
    // if (interpreter->SetNumThreads(num_threads) != kTfLiteOk) {
    //    std::cerr << "Warning: Failed to set TFLite interpreter threads to " << num_threads << std::endl;
    // } else {
    //    std::cerr << "Debug: TFLite interpreter threads set to " << num_threads << std::endl;
    // }


    if (interpreter->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("Failed to allocate tensors for the interpreter.");
    }
    std::cerr << "Debug: Tensors allocated." << std::endl;

    // --- 3. Get Model Input Tensor Details ---
    int input_tensor_idx = interpreter->inputs()[0];
    TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_idx);
    if (!input_tensor) throw std::runtime_error("Failed to get input tensor object.");

    std::cerr << "Debug: Model Input Tensor Details:" << std::endl;
    std::cerr << "  - Index: " << input_tensor_idx << std::endl;
    std::cerr << "  - Type: " << TfLiteTypeGetName(input_tensor->type) << std::endl;
    std::cerr << "  - Dimensions from model: [";
    if (input_tensor->dims) {
        for (int i = 0; i < input_tensor->dims->size; ++i) {
            std::cerr << input_tensor->dims->data[i] << (i < input_tensor->dims->size - 1 ? "," : "");
        }
    } else { std::cerr << "NULL_DIMS"; }
    std::cerr << "]" << std::endl;

    size_t model_expected_input_elements = GetTensorSize(input_tensor->dims);
    std::cerr << "  - Total elements expected by model: " << model_expected_input_elements << std::endl;

    if (model_expected_input_elements == 0) {
        throw std::runtime_error("Model expects 0 input elements.");
    }
    if (input_tensor->dims->size != 4 || input_tensor->dims->data[0] != 1) {
        throw std::runtime_error("Model input tensor must be [1,C,H,W] shape.");
    }
    int model_C = input_tensor->dims->data[1];
    int model_H = input_tensor->dims->data[2];
    int model_W = input_tensor->dims->data[3];

    // --- 4. Validate NPY data against Model Input ---
    if (npy_C != model_C || npy_H != model_H || npy_W != model_W) {
        std::string npy_dims_str = "(" + std::to_string(npy_C) + "," + std::to_string(npy_H) + "," + std::to_string(npy_W) + ")";
        std::string model_dims_str = "(" + std::to_string(model_C) + "," + std::to_string(model_H) + "," + std::to_string(model_W) + ")";
        throw std::runtime_error("NPY sample dimensions " + npy_dims_str +
                                 " (C,H,W after N) do not match model input dimensions " + model_dims_str +
                                 " (C,H,W after B=1). Please regenerate NPY data with correct dimensions (e.g., time length " + std::to_string(model_W) + ").");
    }
    if (elements_per_npy_sample != model_expected_input_elements) {
        // This check is somewhat redundant if the C,H,W dimensions match, but good for sanity.
        throw std::runtime_error("Mismatch between elements per NPY sample (" + std::to_string(elements_per_npy_sample) +
                                 ") and model expected input elements (" + std::to_string(model_expected_input_elements) + "). Check NPY generation.");
    }

    // --- 5. Model Input Type Check ---
    if (input_tensor->type != kTfLiteFloat32) {
        // For dynamic range int8 quantized models, input is typically still Float32.
        // If you have a fully int8 quantized model (input is int8), you'll need to add quantization logic here.
        throw std::runtime_error("This benchmark is set up for Float32 model input. Model expects: " +
                                 std::string(TfLiteTypeGetName(input_tensor->type)));
    }
    float* model_input_tensor_ptr = interpreter->typed_tensor<float>(input_tensor_idx);
    if (!model_input_tensor_ptr && model_expected_input_elements > 0) {
        throw std::runtime_error("Failed to get typed_tensor<float> pointer from interpreter.");
    }

    // --- 6. Warmup Phase ---
    std::cerr << "Debug: Starting warmup..." << std::endl;
    for (int iter = 0; iter < warmup_iterations; ++iter) {
        const float* npy_sample_for_warmup = full_npy_data.data(); // Use the first NPY sample

        if (model_expected_input_elements > 0) {
            std::copy_n(npy_sample_for_warmup, model_expected_input_elements, model_input_tensor_ptr);
        }

        std::cerr << "Debug: Invoking interpreter (warmup iteration " << iter << ")..." << std::endl;
        if (interpreter->Invoke() != kTfLiteOk) {
            throw std::runtime_error("Interpreter invoke failed during warmup iteration " + std::to_string(iter));
        }
        std::cerr << "Debug: Invoke successful (warmup iteration " << iter << ")." << std::endl;
    }
    std::cerr << "Debug: Warmup completed." << std::endl;

    // --- 7. Benchmarking Phase ---
    std::cerr << "Debug: Starting benchmark..." << std::endl;
    auto benchmark_start_time = std::chrono::high_resolution_clock::now();
    struct rusage rusage_before, rusage_after;
    getrusage(RUSAGE_SELF, &rusage_before);

    for (int i = 0; i < num_total_npy_samples; ++i) {
        const float* current_npy_sample_ptr = &full_npy_data[i * elements_per_npy_sample];

        if (model_expected_input_elements > 0) {
            std::copy_n(current_npy_sample_ptr, model_expected_input_elements, model_input_tensor_ptr);
        }

        if (interpreter->Invoke() != kTfLiteOk) {
            throw std::runtime_error("Interpreter invoke failed during benchmark run on NPY sample " + std::to_string(i));
        }
    }

    auto benchmark_end_time = std::chrono::high_resolution_clock::now();
    getrusage(RUSAGE_SELF, &rusage_after);
    std::cerr << "Debug: Benchmark completed." << std::endl;

    // --- 8. Calculate and Store Statistics ---
    double total_duration_ms = std::chrono::duration<double, std::milli>(benchmark_end_time - benchmark_start_time).count();
    double avg_latency_ms = (num_total_npy_samples > 0) ? (total_duration_ms / num_total_npy_samples) : 0.0;
    double user_cpu_time_sec = (rusage_after.ru_utime.tv_sec - rusage_before.ru_utime.tv_sec) +
                               (rusage_after.ru_utime.tv_usec - rusage_before.ru_utime.tv_usec) / 1e6;
    double system_cpu_time_sec = (rusage_after.ru_stime.tv_sec - rusage_before.ru_stime.tv_sec) +
                                 (rusage_after.ru_stime.tv_usec - rusage_before.ru_stime.tv_usec) / 1e6;
    double total_cpu_time_sec = user_cpu_time_sec + system_cpu_time_sec;
    double cpu_usage_percent = (total_duration_ms > 0.001) ? (total_cpu_time_sec / (total_duration_ms / 1000.0)) * 100.0 : 0.0;
    double ram_usage_mb = static_cast<double>(rusage_after.ru_maxrss) / 1024.0; // Kilobytes to Megabytes
    struct stat model_stat;
    if (stat(model_path.c_str(), &model_stat) != 0) {
        std::cerr << "Warning: Could not stat model file: " << model_path << std::endl;
        model_stat.st_size = 0;
    }
    double model_size_mb = static_cast<double>(model_stat.st_size) / (1024.0 * 1024.0);

    std::cerr << "Debug: Results for " << model_path << ":" << std::endl;
    std::cerr << "  - Total inference time for " << num_total_npy_samples << " samples: " << std::fixed << std::setprecision(2) << total_duration_ms << " ms" << std::endl;
    std::cerr << "  - Average latency: " << avg_latency_ms << " ms/sample" << std::endl;
    std::cerr << "  - Total CPU time (user + system): " << total_cpu_time_sec << " s" << std::endl;
    std::cerr << "  - CPU usage during benchmark: " << std::fixed << std::setprecision(1) << cpu_usage_percent << "%" << std::endl;
    std::cerr << "  - Peak RAM usage (Max RSS): " << ram_usage_mb << " MB" << std::endl;
    std::cerr << "  - Model file size: " << model_size_mb << " MB" << std::endl;

    return {avg_latency_ms, cpu_usage_percent, ram_usage_mb, model_size_mb};
}


// --- Main function ---
int main(int argc, char** argv) {
    std::vector<std::string> model_paths;
    std::string npy_file_path;
    int warmup_iters = 1;

    // Basic argument parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--models") {
            if (i + 1 < argc) {
                while (i + 1 < argc && argv[i+1][0] != '-') { // Basic check for next arg not being an option
                    model_paths.push_back(argv[++i]);
                }
            } else {
                std::cerr << "Error: --models option requires at least one model path." << std::endl; return 1;
            }
        } else if (arg == "--input") {
            if (i + 1 < argc) {
                npy_file_path = argv[++i];
            } else {
                std::cerr << "Error: --input option requires a path to an NPY file." << std::endl; return 1;
            }
        } else if (arg == "--warmup") {
            if (i + 1 < argc) {
                try {
                    warmup_iters = std::stoi(argv[++i]);
                    if (warmup_iters < 0) warmup_iters = 0; // Treat negative as 0
                } catch (const std::exception& e) {
                    std::cerr << "Error: Invalid warmup count '" << argv[i] << "'. Using default 1. (" << e.what() << ")" << std::endl;
                    warmup_iters = 1;
                }
            } else {
                std::cerr << "Error: --warmup option requires a number." << std::endl; return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            // Fall through to print usage
        }
    }

    if (model_paths.empty() || npy_file_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " --models <model1.tflite> [model2.tflite ...] --input <sample_data.npy> [--warmup <N>]" << std::endl;
        return 1;
    }

    // Load NPY data once
    std::vector<int> npy_shape;
    std::vector<float> npy_data;
    try {
        npy_data = load_npy(npy_file_path, npy_shape);
    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: Could not load NPY file '" << npy_file_path << "': " << e.what() << std::endl;
        return 1;
    }

    // Print Markdown table header
    std::cout << "| Model Path | Avg Latency (ms) | CPU Usage (%) | Peak RAM (MB) | Model Size (MB) |\n";
    std::cout << "|------------|------------------|---------------|---------------|-----------------|\n";

    // Process each model
    for (const auto& model_file : model_paths) {
        std::string model_name = model_file;
        // Extract filename from path for cleaner table output
        size_t last_slash_idx = model_file.find_last_of("/\\");
        if (std::string::npos != last_slash_idx) {
            model_name = model_file.substr(last_slash_idx + 1);
        }

        try {
            BenchmarkStats stats = measure_model_performance(model_file, npy_data, npy_shape, warmup_iters);
            std::cout << "| " << model_name
                      << " | " << std::fixed << std::setprecision(2) << stats.avg_latency_ms
                      << " | " << std::fixed << std::setprecision(1) << stats.cpu_usage_percent
                      << " | " << std::fixed << std::setprecision(1) << stats.ram_usage_mb
                      << " | " << std::fixed << std::setprecision(2) << stats.model_size_mb
                      << " |\n";
        } catch (const std::exception& e) {
            std::cerr << "Error processing model '" << model_file << "': " << e.what() << std::endl;
            std::cout << "| " << model_name
                      << " | ERROR | ERROR | ERROR | ERROR |\n";
        }
    }

    return 0;
}