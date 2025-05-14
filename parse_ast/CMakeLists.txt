cmake_minimum_required(VERSION 3.5)

project(parse_ast)
set(CMAKE_CXX_STANDARD 17)

set(HOME $ENV{HOME})
set(TAICHI_DIR ${HOME}/git/taichi)
message(STATUS "TAICHI_DIR: ${TAICHI_DIR}")

set(PROJECT_SOURCE_DIR ${TAICHI_DIR})

set(EXT_INCLUDES /Users/hugh/git/taichi/external/spdlog/include
${PROJECT_SOURCE_DIR}/external/spdlog/include
${PROJECT_SOURCE_DIR}/external/eigen
${PROJECT_SOURCE_DIR}/external/volk
${PROJECT_SOURCE_DIR}/external/SPIRV-Tools/include
${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include
${PROJECT_SOURCE_DIR}/external/imgui
${PROJECT_SOURCE_DIR}/external/imgui/backends
${PROJECT_SOURCE_DIR}/external/FP16/include
)
set(TI_LIBRARIES
${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/libtaichi_core_static.a
${TAICHI_DIR}//_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/runtime/cpu/libcpu_runtime.a
${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/common/libtaichi_common.a
${TAICHI_DIR}//_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/compilation_manager/libcompilation_manager.a
${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/rhi/libti_device_api.a
${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/rhi/metal/libmetal_rhi.a
${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/rhi/cpu/libcpu_rhi.a
${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/rhi/interop/libinterop_rhi.a
# ${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/rhi/llvm/libllvm_rhi.a
${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/rhi/common/libcommon_rhi.a
${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/codegen/cpu/libcpu_codegen.a
# ${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/codegen/llvm/libllvm_codegen.a
${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/util/libtaichi_util.a
${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/runtime/program_impls/metal/libmetal_program_impl.a
${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/runtime/gfx/libgfx_runtime.a
# libtaichi_core_static.a  
${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/codegen/cpu/libcpu_codegen.a  ${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/runtime/program_impls/llvm/libllvm_program_impl.a  ${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/runtime/cpu/libcpu_runtime.a  ${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/codegen/llvm/libllvm_codegen.a  ${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/runtime/llvm/libllvm_runtime.a  ${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/rhi/libti_device_api.a  ${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/rhi/metal/libmetal_rhi.a  ${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/external/SPIRV-Cross/libspirv-cross-msl.a  ${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/external/SPIRV-Cross/libspirv-cross-glsl.a  ${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/external/SPIRV-Cross/libspirv-cross-core.a   ${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/rhi/cpu/libcpu_rhi.a  ${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/rhi/llvm/libllvm_rhi.a  ${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/rhi/interop/libinterop_rhi.a  ${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/rhi/common/libcommon_rhi.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMOrcJIT.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMExecutionEngine.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMJITLink.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMOrcTargetProcess.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMOrcShared.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMRuntimeDyld.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMPasses.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMCoroutines.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMObjCARCOpts.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMipo.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMLinker.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMVectorize.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMInstrumentation.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMFrontendOpenMP.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMIRReader.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMAsmParser.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMAArch64CodeGen.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMAsmPrinter.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMGlobalISel.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMSelectionDAG.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMCodeGen.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMBitWriter.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMScalarOpts.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMInstCombine.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMAggressiveInstCombine.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMTransformUtils.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMTarget.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMAnalysis.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMProfileData.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMSymbolize.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMDebugInfoPDB.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMDebugInfoMSF.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMDebugInfoDWARF.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMObject.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMBitReader.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMTextAPI.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMCFGuard.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMCore.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMRemarks.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMBitstreamReader.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMAArch64AsmParser.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMMCParser.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMAArch64Disassembler.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMAArch64Desc.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMMCDisassembler.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMAArch64Info.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMMC.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMBinaryFormat.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMDebugInfoCodeView.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMAArch64Utils.a  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMSupport.a  -lm  /Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk/usr/lib/libz.tbd  /Users/hugh/.cache/ti-build-cache/llvm15-m1-nozstd/lib/libLLVMDemangle.a  ${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/runtime/program_impls/metal/libmetal_program_impl.a  ${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/runtime/program_impls/gfx/libgfx_program_impl.a  ${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/compilation_manager/libcompilation_manager.a  ${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/codegen/spirv/libspirv_codegen.a  ${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/util/libtaichi_util.a  ${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/external/SPIRV-Tools/source/opt/libSPIRV-Tools-opt.a  ${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/external/SPIRV-Tools/source/libSPIRV-Tools.a  ${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/runtime/gfx/libgfx_runtime.a  ${TAICHI_DIR}/_skbuild/macosx-15.0-arm64-3.10/cmake-build/taichi/common/libtaichi_common.a  "-framework Foundation"  "-framework Metal"  "-framework ApplicationServices"  "-framework Cocoa" "-framework QuartzCore"
)

add_definitions(-DTI_INCLUDED)


add_executable(test_dump_load_whole_collider
    test_dump_load_whole_collider.cpp
)
target_include_directories(test_dump_load_whole_collider PRIVATE
    ${TAICHI_DIR}
    ${EXT_INCLUDES}
)
target_link_libraries(test_dump_load_whole_collider PRIVATE
   ${TI_LIBRARIES}
)

add_executable(parse_ast_1
    parse_ast_1.cpp
)
target_include_directories(parse_ast_1 PRIVATE
    ${TAICHI_DIR}
    ${EXT_INCLUDES}
)
target_link_libraries(parse_ast_1 PRIVATE
   ${TI_LIBRARIES}
)
