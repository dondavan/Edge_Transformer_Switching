#include "src/gpu/cl/operators/ClLinear.h"

#include "src/common/utils/Log.h"
#include "src/gpu/cl/ClCompileContext.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "src/gpu/cl/utils/ClAuxTensorHandler.h"

#include "src/gpu/cl/kernels/ClLinearKernel.h"
#include "src/runtime/heuristics/matmul_native/ClMatMulNativeKernelConfig.h"
#include "src/runtime/heuristics/matmul_native/IClMatMulNativeKernelConfig.h"

namespace arm_compute
{
namespace opencl
{
void ClLinear::configure(const ClCompileContext &compile_context,
                         ITensorInfo            *a,
                         ITensorInfo            *b,
                         ITensorInfo            *c,
                         ITensorInfo            *d,
                         float                   alpha,
                         float                   beta,
                         const LinearLayerInfo  &linear_info)
{
    ARM_COMPUTE_LOG_PARAMS(a, b, c, d, alpha, beta, linear_info);
    ARM_COMPUTE_UNUSED(a, b, c, d, alpha);
    ARM_COMPUTE_UNUSED(linear_info);
    ARM_COMPUTE_UNUSED(beta);

    // Specify whether transpose weights is necessary in matmul info
    const MatMulInfo mat_info = MatMulInfo().adj_rhs(true);

    const GPUTarget                                         gpu_target = CLScheduler::get().target();
    std::unique_ptr<cl_matmul::IClMatMulNativeKernelConfig> kernel_config =
        cl_matmul::ClMatMulNativeKernelConfigurationFactory::create(gpu_target);
    MatMulKernelInfo kernel_info = kernel_config->configure(a, b, mat_info);

    auto k = std::make_unique<kernels::ClLinearKernel>();
    k->set_target(gpu_target);
    k->configure(compile_context, a, b, c, d, alpha, beta, kernel_info);
    _kernel = std::move(k);
}

Status
ClLinear::validate(const ITensorInfo *a,
                   const ITensorInfo *b,
                   const ITensorInfo *c,
                   ITensorInfo       *d,
                   float              alpha,
                   float beta, const LinearLayerInfo &linear_info)
{
    ARM_COMPUTE_UNUSED(a);
    ARM_COMPUTE_UNUSED(b);
    ARM_COMPUTE_UNUSED(c);
    ARM_COMPUTE_UNUSED(d);
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(beta);
    ARM_COMPUTE_UNUSED(linear_info);
    return Status{};
}

void ClLinear::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
    CLScheduler::get().enqueue_op(*_kernel.get(), tensors, true);
}

} // namespace opencl
} // namespace arm_compute
