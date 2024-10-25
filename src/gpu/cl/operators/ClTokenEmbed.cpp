#include "src/gpu/cl/operators/ClTokenEmbed.h"

#include "arm_compute/runtime/CL/CLScheduler.h"

#include "src/common/IOperator.h"
#include "src/common/utils/LegacySupport.h"
#include "src/common/utils/Log.h"
#include "src/gpu/cl/kernels/ClVectorizeKernel.h"

namespace arm_compute
{
namespace opencl
{
void ClTokenEmbed::configure(const ClCompileContext   &compile_context,
                             const ITensorInfo        *input,
                             const ITensorInfo        *vocab,
                             ITensorInfo              *output,
                             const EmbeddingLayerInfo &tkemb_info)
{
    ARM_COMPUTE_LOG_PARAMS(input, output, tkemb_info);
    ARM_COMPUTE_UNUSED(tkemb_info);

    auto k = std::make_unique<kernels::ClVectorizeKernel>();
    k->configure(compile_context, input, vocab, output);
    _kernel = std::move(k);

}

Status
ClTokenEmbed::validate(const ITensorInfo *input, const ITensorInfo *vocab, const ITensorInfo *output, const EmbeddingLayerInfo &tkemb_info)
{
    ARM_COMPUTE_UNUSED(input);
    ARM_COMPUTE_UNUSED(vocab);
    ARM_COMPUTE_UNUSED(output);
    ARM_COMPUTE_UNUSED(tkemb_info);
    return Status{};
}

void ClTokenEmbed::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");

    CLScheduler::get().enqueue_op(*_kernel.get(), tensors);

}

} // namespace opencl
} // namespace arm_compute
