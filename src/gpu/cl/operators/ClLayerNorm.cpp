#include "src/gpu/cl/operators/ClLayerNorm.h"

#include "src/common/utils/Log.h"
#include "src/gpu/cl/ClCompileContext.h"

#include "src/gpu/cl/kernels/ClLayerNormKernel.h"

namespace arm_compute
{
namespace opencl
{
void ClLayerNorm::configure(const CLCompileContext   &compile_context,
                            const ITensorInfo        *input,
                            ITensorInfo              *output,
                            const LayerNormLayerInfo &info)
{
    auto k = std::make_unique<kernels::ClLayerNormKernel>();
    k->configure(compile_context, input, output, info);
    _layer_norm_kernel = std::move(k); 
}

Status
ClLayerNorm::validate(const ITensorInfo        *input,
                      ITensorInfo              *output,
                      const LayerNormLayerInfo &info)
{
    ARM_COMPUTE_UNUSED(input);
    ARM_COMPUTE_UNUSED(output);
    ARM_COMPUTE_UNUSED(info);
    return Status{};
}

void ClLayerNorm::run(ITensorPack &tensors)
{
    // Run indirect convolution
    CLScheduler::get().enqueue_op(*_layer_norm_kernel.get(), tensors, true);
}

} // namespace opencl
} // namespace arm_compute
