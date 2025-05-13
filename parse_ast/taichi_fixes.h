#pragma once

// This header provides fixes for namespace issues when using Taichi headers
#include <string>

namespace taichi {

// Forward declarations to prevent namespace issues
namespace zip {
std::vector<uint8_t> read(const std::string &filename);
void write(const std::string &filename, uint8_t *data, std::size_t size);
}

// Add namespace fixes
#define ends_with taichi::ends_with
#define zip taichi::zip

}  // namespace taichi
