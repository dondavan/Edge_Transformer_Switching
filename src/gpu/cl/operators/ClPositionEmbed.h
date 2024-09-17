#ifndef ARM_COMPUTE_CL_POSITION_EMBED_H
#define ARM_COMPUTE_CL_POSITION_EMBED_H

#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClOperator.h"

namespace arm_compute
{
namespace opencl
{
/** Basic function to run @ref kernels::CpuEmbedKernel */
class ClPositionEmbed : public IClOperator
{
    public:
    /** Configure operator for a given list of arguments
     *
     * @param[in] compile_context The compile context to be used.
     * @param[in]  input           Source tensor info. Data types supported: U8.
     * @param[in]  position         Const Position vector, Data type supported: F32
     * @param[out] output          Destination tensor info. Data type supported: F32
     */
    void configure(const ClCompileContext &compile_context,
                   const ITensorInfo      *input,
                   const ITensorInfo      *position,
                   ITensorInfo            *output);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuPositionEmbed::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *position, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
};
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_POSITION_EMBED_H */
