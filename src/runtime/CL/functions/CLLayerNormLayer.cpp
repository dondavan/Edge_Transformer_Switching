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
    const ITensor                     *src{ nullptr };
    ITensor                           *dst{ nullptr };
    std::unique_ptr<opencl::ClLayerNorm> op{ nullptr };
};

CLLayerNormLayer::CLLayerNormLayer()
    : _impl(std::make_unique<Impl>())
{
}
CLLayerNormLayer::~CLLayerNormLayer() = default;

void CLLayerNormLayer::configure(const ITensor          *input,
                                 ITensor                *output,
                                 const LayerNormLayerInfo &LayerNorm_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, LayerNorm_info);
}
void CLLayerNormLayer::configure(const CLCompileContext   &compile_context,
                                 const ITensor          *input,
                                 ITensor                *output,
                                 const LayerNormLayerInfo &LayerNorm_info)
{
    /*
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

#ifdef MEASURE_TIME
    auto          end_time  = std::chrono::high_resolution_clock::now();
    double        cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt", std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "CLLayerNormLayer::configure cost: " << cost_time << std::endl;
    measure_out.close();
#endif
    */
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_LOG_PARAMS(input, output);

    _impl->src = input;
    _impl->dst = output;

    _impl->op = std::make_unique<opencl::ClLayerNorm>();
    _impl->op->configure(compile_context, input->info(), output->info(), LayerNorm_info);
}

Status CLLayerNormLayer::validate(const ITensor          *input,
                                  ITensor                *output,
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
    measure_out << std::scientific << "output x " << _impl->dst->info()->tensor_shape().x() << std::endl;
    measure_out << std::scientific << "output y " << _impl->dst->info()->tensor_shape().y()<< std::endl;
    measure_out << std::scientific << "output z " << _impl->dst->info()->tensor_shape().z()<< std::endl;
    measure_out.close();
#endif
}

} // namespace arm_compute
