#include "arm_compute/runtime/CL/functions/CLLayerNormLayer.h"

#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTypes.h"

#include "src/common/utils/Log.h"

#include "src/gpu/cl/operators/ClLayerNorm.h"

#ifdef MEASURE_TIME
#include <chrono>
#include <fstream>
#endif

namespace arm_compute
{

struct CLLayerNormLayer::Impl
{
    const ICLTensor                     *src{ nullptr };
    ICLTensor                           *dst{ nullptr };
    std::unique_ptr<opencl::ClLayerNorm> op{ nullptr };
};

CLLayerNormLayer::CLLayerNormLayer()
    : _impl(std::make_unique<Impl>())
{
}
CLLayerNormLayer::~CLLayerNormLayer() = default;

void CLLayerNormLayer::configure(const ICLTensor          *input,
                                 ICLTensor                *output,
                                 const LayerNormLayerInfo &LayerNorm_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, LayerNorm_info);
}
void CLLayerNormLayer::configure(const CLCompileContext   &compile_context,
                                 const ICLTensor          *input,
                                 ICLTensor                *output,
                                 const LayerNormLayerInfo &LayerNorm_info)
{
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif


    std::cout << "CLLayerNormLayer::configure start" << std::endl;
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_LOG_PARAMS(input, output);

    _impl->src = input;
    _impl->dst = output;

    _impl->op = std::make_unique<opencl::ClLayerNorm>();
    _impl->op->configure(compile_context, input->info(), output->info(), LayerNorm_info);

    std::cout << "CLLayerNormLayer::configure end" << std::endl;

#ifdef MEASURE_TIME
    auto          end_time  = std::chrono::high_resolution_clock::now();
    double        cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt", std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "CLLayerNormLayer::configure cost: " << cost_time << std::endl;
    measure_out.close();
#endif
}

Status CLLayerNormLayer::validate(const ICLTensor          *input,
                                  ICLTensor                *output,
                                  const LayerNormLayerInfo &LayerNorm_info)
{
    ARM_COMPUTE_UNUSED(LayerNorm_info);
    return opencl::ClLayerNorm::validate(input->info(), output->info(), LayerNorm_info);
}

void CLLayerNormLayer::run()
{
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

    ITensorPack pack;

    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);

    _impl->op->run(pack);

#ifdef MEASURE_TIME
    auto          end_time  = std::chrono::high_resolution_clock::now();
    double        cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt", std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "CLLayerNormLayer::run cost: " << cost_time << std::endl;
    measure_out.close();
#endif
}

} // namespace arm_compute
