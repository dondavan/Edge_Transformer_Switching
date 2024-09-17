#ifndef SRC_CPU_KERNELS_CL_Linear_KERNEL_H
#define SRC_CPU_KERNELS_CL_Linear_KERNEL_H

#include "src/core/common/Macros.h"

#include "arm_compute/core/KernelDescriptors.h"

#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
/** Interface for the vectorization kernel */
class ClLinearKernel : public IClKernel
{
    public:
    /* Default Constructor */
    ClLinearKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClLinearKernel);
    /** Configure kernel for a given list of arguments
     *
     * @param[in]   src             Source tensor info. Data types supported: U8.
     * @param[in]   vector          Const target vector tensor info, Data type supported: F32
     * @param[out]  dst             Destination tensor info. Data type supported: F32
     * @param[in]   tkemb_info      Token embedding layer information.
     */
    void configure(const CLCompileContext &compile_context,
                   ITensorInfo      *a,
                   ITensorInfo      *b,
                   ITensorInfo      *c,
                   ITensorInfo            *d,
                   float                   alpha,
                   float                   beta,
                   const MatMulKernelInfo &matmul_kernel_info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to ClLinearKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *vector, ITensorInfo *dst);

    // Inherited methods overridden:
    //void        run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;

    private:
    int _m{1};
    int _n{1};
    int _k{1};
};

} // namespace kernels
} // namespace opencl
} // namespace arm_compute

#endif /* SRC_CPU_KERNELS_CL_Linear_KERNEL_H */