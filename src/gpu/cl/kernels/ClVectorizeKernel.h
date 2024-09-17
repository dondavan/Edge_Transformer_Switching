#ifndef SRC_CPU_KERNELS_CL_VECTORIZE_KERNEL_H
#define SRC_CPU_KERNELS_CL_VECTORIZE_KERNEL_H

#include "src/core/common/Macros.h"

#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
/** Interface for the vectorization kernel */
class ClVectorizeKernel : public IClKernel
{
    private:
    using VectorizeKernelPtr =
        std::add_pointer<void(const ITensor *, const ITensor *, ITensor *, const Window &)>::type;

    public:
    /* Default Constructor */
    ClVectorizeKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClVectorizeKernel);
    /** Configure kernel for a given list of arguments
     *
     * @param[in]   src             Source tensor info. Data types supported: U8.
     * @param[in]   vector          Const target vector tensor info, Data type supported: F32
     * @param[out]  dst             Destination tensor info. Data type supported: F32
     * @param[in]   tkemb_info      Token embedding layer information.
     */
    void configure(const CLCompileContext &compile_context, const ITensorInfo *src, const ITensorInfo *vector, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to ClVectorizeKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *vector, ITensorInfo *dst);

    // Inherited methods overridden:
    //void        run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;

};

} // namespace kernels
} // namespace opencl
} // namespace arm_compute

#endif /* SRC_CPU_KERNELS_CL_VECTORIZE_KERNEL_H */