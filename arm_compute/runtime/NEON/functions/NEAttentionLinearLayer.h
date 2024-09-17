#ifndef ARM_COMPUTE_NEATTENTIONLINEARLAYER_H
#define ARM_COMPUTE_NEATTENTIONLINEARLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IRuntimeContext.h"
#include "arm_compute/runtime/Tensor.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class ITensor;
class ITensorInfo;

class NEAttentionLinearLayer : public IFunction
{
    public:
    /** Default Constructor */
    NEAttentionLinearLayer();
    /** Default Destructor */
    ~NEAttentionLinearLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEAttentionLinearLayer(const NEAttentionLinearLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEAttentionLinearLayer &operator=(const NEAttentionLinearLayer &) = delete;

    /** Set the input and output tensor.
     * 
     * @param[in]  query      Input tenser of Attention Query, Data type supported: F32
     * @param[in]  key        Input tensor of Attention Key, Data type supported: F32
     * @param[in]  value      Input tenser of Attention Value, Data type supported: F32
     * @param[out] output     Output tensor, shape (d_model,d_model). Data type supported: F32
     */
    void configure(const ITensor *query_input, const ITensor *query_w, const ITensor *query_b,
                   const ITensor *key_input, const ITensor *key_w, const ITensor *key_b,
                   const ITensor *value_input, const ITensor *value_w, const ITensor *value_b,
                   ITensor *query_output, ITensor *key_output, ITensor *value_output,
                   const LinearLayerInfo& linear_info);
    /** Static function to check if given info will lead to a valid configuration of @ref NEAttentionLinearLayer
     *
     * @param[in] output Destination tensor info. Data type supported: same as @p input
     *
     * @return a status
     */
    static Status validate(ITensor *output);

    // Inherited methods overridden:
    void run() override;

    private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute

#endif /* ARM_COMPUTE_NEATTENTIONLINEARLAYER_H */