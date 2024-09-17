#ifndef ARM_COMPUTE_CL_LINEAR_H
#define ARM_COMPUTE_CL_LINEAR_H

#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"
#include "src/gpu/cl/IClOperator.h"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"


namespace arm_compute
{
namespace opencl
{

/** Basic function to run @ref kernels::CpuLinearKernel 
 * @note Performs linear function [alpha * A * B + beta * C]
*/
class ClLinear : public IClOperator
{
public:
    /** Initialise the kernel's inputs and output
     *
     * @param[in]  a      An input tensor. Data type supported: f32.
     * @param[in]  b      An input tensor. Data type supported: f32.
     * @param[in]  c      An input bias ensor. Data type supported: f32.
     * @param[out] d      Output tensor. Data type supported: F32.
     * @param[in]  alpha  Weight of the matrix product
     * @param[in]  beta   Weight of matrix C
     * @param[in]  info   (Optional)Linear layer operation information
     */
    void configure(const ClCompileContext &compile_context,
                   ITensorInfo *a,
                   ITensorInfo *b,
                   ITensorInfo *c,
                   ITensorInfo       *d,
                   float              alpha,
                   float              beta, 
                   const LinearLayerInfo& info = LinearLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CpuLinearKernel
     *
     * Similar to @ref CpuGemm::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *a,
                           const ITensorInfo *b,
                           const ITensorInfo *c,
                           ITensorInfo       *d,
                           float              alpha,
                           float              beta,
                           const LinearLayerInfo& info = LinearLayerInfo());

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;

private:
    
};

} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_LINEAR_H */
