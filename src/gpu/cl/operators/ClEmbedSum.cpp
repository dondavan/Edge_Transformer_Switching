#include "src/gpu/cl/operators/ClEmbedSum.h"

#include "src/common/utils/Log.h"
#include "src/gpu/cl/ClCompileContext.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "src/gpu/cl/utils/ClAuxTensorHandler.h"

namespace arm_compute
{
namespace opencl
{
void ClEmbedSum::configure(const ClCompileContext   &compile_context,
                           ITensorInfo              *token,
                           ITensorInfo              *segemnt,
                           ITensorInfo              *position,
                           ITensorInfo              *output,
                           const EmbeddingLayerInfo &emb_info)
{
    ARM_COMPUTE_UNUSED(emb_info);
    
    auto k = std::make_unique<kernels::ClEmbSumKernel>();
    k->configure(compile_context,  token, segemnt,position, output);
    _kernel = std::move(k);

}

Status
ClEmbedSum::validate(const ITensorInfo        *token,
                     const ITensorInfo        *segemnt,
                     const ITensorInfo        *position,
                     ITensorInfo              *output,
                     const EmbeddingLayerInfo &emb_info)
{
    ARM_COMPUTE_UNUSED(token);
    ARM_COMPUTE_UNUSED(segemnt);
    ARM_COMPUTE_UNUSED(position);
    ARM_COMPUTE_UNUSED(output);
    ARM_COMPUTE_UNUSED(emb_info);
    return Status{};
}



} // namespace opencl
} // namespace arm_compute
