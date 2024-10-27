#ifndef ARM_COMPUTE_CL_ATTENTION_LINEAR_LAYER_H
#define ARM_COMPUTE_CL_ATTENTION_LINEAR_LAYER_H

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Types.h"

#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IWeightsManager.h"
#include "arm_compute/runtime/MemoryGroup.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class CLCompileContext;
class ITensor;
class ITensorInfo;

/** Perform basic linear function */
class CLAttentionLinearLayer : public IFunction
{
    public:
    /** Constructor */
    CLAttentionLinearLayer(std::shared_ptr<IMemoryManager> memory_manager  = nullptr,
                           IWeightsManager                *weights_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLAttentionLinearLayer(const CLAttentionLinearLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    CLAttentionLinearLayer(CLAttentionLinearLayer &&) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLAttentionLinearLayer &operator=(const CLAttentionLinearLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    CLAttentionLinearLayer &operator=(CLAttentionLinearLayer &&) = delete;
    /** Destructor */
    ~CLAttentionLinearLayer();

    /** Initialise the kernel's inputs and output
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0           |dst          |
     * |:--------------|:------------|
     * |F32            |F32          |
     *
     * @param[in]  input1 First tensor input. Data type supported: F32.
     * @param[out] output Output tensor. Data type supported: F32.
     */
    void configure(const ITensor *query_input, const ITensor *query_w, const ITensor *query_b,
                   const ITensor *key_input, const ITensor *key_w, const ITensor *key_b,
                   const ITensor *value_input, const ITensor *value_w, const ITensor *value_b,
                   ITensor *query_output, ITensor *key_output, ITensor *value_output,
                   const LinearLayerInfo& linear_info);
    /** Initialise the kernel's inputs and output
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0           |dst          |
     * |:--------------|:------------|
     * |F32            |F32          |
     *
     * @param[in]  input1 First tensor input. Data type supported: F32.
     * @param[out] output Output tensor. Data type supported: F32.
     */
    void configure(const CLCompileContext &compile_context,
                   const ITensor *query_input, const ITensor *query_w, const ITensor *query_b,
                   const ITensor *key_input, const ITensor *key_w, const ITensor *key_b,
                   const ITensor *value_input, const ITensor *value_w, const ITensor *value_b,
                   ITensor *query_output, ITensor *key_output, ITensor *value_output,
                   const LinearLayerInfo &linear_info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLAttentionLinearLayer
     *
     * @param[in] input1 First input tensor info. Data types supported: F32.
     * @param[in] output Output tensor info. Data type supported: F32.
     *
     * @return a status
     */
    static Status validate(const ITensor *input, const ITensor *weight, const ITensor *bias, ITensor *output, const LinearLayerInfo &linear_info);

    // Inherited methods overridden
    void run() override;

    private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace arm_compute

#endif /* ARM_COMPUTE_CL_ATTENTION_LINEAR_LAYER_H */