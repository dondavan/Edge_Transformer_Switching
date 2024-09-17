#ifndef ARM_COMPUTE_CL_POSITION_EMBEDDING_KERNEL_H
#define ARM_COMPUTE_CL_POSITION_EMBEDDING_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
/** Kernel to perform tensor position Embedding */
class ClPositionEmbeddingKernel : public IClKernel
{
public:
    /** Default constructor */
    ClPositionEmbeddingKernel() = default;
    /** Default destructor */
    ~ClPositionEmbeddingKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClPositionEmbeddingKernel);
    /** Configure kernel for a given list of arguments
     *
     * @note Arbitrary permutation vectors are supported with rank not greater than 4
     *
     * @param[in] compile_context The compile context to be used.
     * @param[in]  src  Srouce tensor to permute. Data types supported: All
     * @param[in]  pos  Pretrained position embedding. Data types supported: All
     * @param[out] dst  Destination tensor. Data types supported: Same as @p src
     * @param[in]  perm Permutation vector
     */
    void configure(const CLCompileContext  &compile_context,
                   const ITensorInfo *src, 
                   const ITensorInfo *pos, 
                   ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClPositionEmbeddingKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *pos, const ITensorInfo *dst);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;

private:
    unsigned int _d_model{512U};
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_POSITION_EMBEDDING_KERNEL_H */
