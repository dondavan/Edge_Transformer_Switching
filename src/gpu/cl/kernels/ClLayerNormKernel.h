#ifndef ARM_COMPUTE_CLLAYERNORMKERNEL_H
#define ARM_COMPUTE_CLLAYERNORMKERNEL_H

#include "src/core/common/Macros.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
/** Interface for the kernel to perform layer normalization */
class ClLayerNormKernel : public IClKernel
{
    private:
    using LayerNormKernelPtr =
        std::add_pointer<void(const ITensor *, ITensor *, const LayerNormLayerInfo &, const Window &)>::type;

    public:
    /* Default Constructor */
    ClLayerNormKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClLayerNormKernel);

    /** Initialise the kernel's inputs and output
     *
     * @param[in]  input1 An input tensor. Data type supported: f32.
     * @param[out] output Output tensor. Data type supported: F32.
     * @param[out] op     Logical operation to perform
     */
    void configure(const ClCompileContext &compile_context,
                   const ITensorInfo      *input,
                   ITensorInfo            *output,
                   LayerNormLayerInfo      info);
    /** Static function to check if given info will lead to a valid configuration of @ref ClLayerNormKernel
     *
     * @param[in] input An input tensor. Data type supported: F32.
     * @param[in] output Output tensor. Data type supported: F32..
     * @param[in] op     Logical operation to perform
     *
     * @return a status
     */
    static Status
    validate(const ITensorInfo *input, const ITensorInfo *output, LayerNormLayerInfo info);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;

    private:
    const ITensorInfo *_input{ nullptr };
    ITensorInfo       *_output{ nullptr };
    LayerNormLayerInfo _info{};
    LayerNormKernelPtr _run_method{ nullptr };
    std::string        _name{};
};

} // namespace kernels
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLLAYERNORMKERNEL_H */
