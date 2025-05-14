#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <chrono>

// #include "taichi_fixes.h"

#include "taichi/common/core.h"
// #include "taichi/common/types.h"
#include "taichi/common/zip.h"
#include "taichi/common/serialization.h"
#include "taichi/common/json_serde.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/rhi/arch.h"

using namespace taichi;
using namespace taichi::lang;
using namespace liong::json;

int main(int argc, char* argv[]) {
    std::string filename = "/tmp/ast.json";
    
    // If a filename is provided as argument, use it instead
    if (argc > 1) {
        filename = argv[1];
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Read the JSON file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return 1;
    }
    std::string json_content((std::istreambuf_iterator<char>(file)),
    std::istreambuf_iterator<char>());

    // Parse JSON
    JsonValue json_ast = liong::json::parse(json_content);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = (std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time)).count() / 1000000.0;
    std::cout << "elapsed time " << elapsed_time << "s" << std::endl;

    std::cout << "Successfully parsed JSON" << std::endl;

    return 0;
}
