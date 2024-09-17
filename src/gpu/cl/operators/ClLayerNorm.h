#ifndef ARM_COMPUTE_CL_LAYER_NORM_H
#define ARM_COMPUTE_CL_LAYER_NORM_H

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTypes.h"

#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"
#include "src/gpu/cl/IClOperator.h"

namespace arm_compute
{
namespace opencl
{

/** Basic function to run @ref kernels::ClLayerNormKernel 
 * @note Performs LayerNorm function [alpha * A * B + beta * C]
*/
class ClLayerNorm : public IClOperator
{
    public:
    /** Initialise the kernel's inputs and output
     *
     * @param[in]  input      Input tensor. Data type supported: f32.
     * @param[out] output     Output tensor. Data type supported: F32.
     * @param[in]  info       (Optional)LayerNorm layer operation information
     */
    void configure(const CLCompileContext   &compile_context,
                   const ITensorInfo        *input,
                   ITensorInfo              *output,
                   const LayerNormLayerInfo &info = LayerNormLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref ClLayerNormKernel
     *
     * Similar to @ref CpuGemm::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo        *input,
                           ITensorInfo              *output,
                           const LayerNormLayerInfo &info = LayerNormLayerInfo());

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;

    private:
    std::unique_ptr<IClKernel> _layer_norm_kernel{ nullptr };
};

} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_LAYER_NORM_H */
