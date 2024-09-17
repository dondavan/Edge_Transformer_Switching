#ifndef ARM_COMPUTE_CLSCALEDOTPRODUCTIONATTENTIONLAYER_H
#define ARM_COMPUTE_CLSCALEDOTPRODUCTIONATTENTIONLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{

// Forward declarations
class CLCompileContext;
class ICLTensor;
class ICLTensorInfo;

class CLScaleDotProductionAttentionLayer : public IFunction
{
    public:
    /** Default Constructor */
    CLScaleDotProductionAttentionLayer();
    /** Default Destructor */
    ~CLScaleDotProductionAttentionLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLScaleDotProductionAttentionLayer(const CLScaleDotProductionAttentionLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLScaleDotProductionAttentionLayer &operator=(const CLScaleDotProductionAttentionLayer &) = delete;

    /** Set the input and output tensor.
     * 
     * @param[in]  query      Input tenser of Attention Query, Data type supported: F32
     * @param[in]  key        Input tensor of Attention Key, Data type supported: F32
     * @param[in]  value      Input tenser of Attention Value, Data type supported: F32
     * @param[out] output     Output tensor, shape (d_model,d_model). Data type supported: F32
     */
    void configure(const ICLTensor                            *query,
                   const ICLTensor                            *key,
                   const ICLTensor                            *value,
                   ICLTensor                                  *output,
                   const ScaleDotProductionAttentionLayerInfo &info);
    /** Set the input and output tensor.
     * 
     * @param[in]  query      Input tenser of Attention Query, Data type supported: F32
     * @param[in]  key        Input tensor of Attention Key, Data type supported: F32
     * @param[in]  value      Input tenser of Attention Value, Data type supported: F32
     * @param[out] output     Output tensor, shape (d_model,d_model). Data type supported: F32
     */
    void configure(const CLCompileContext                     &compile_context,
                   const ICLTensor                            *query,
                   const ICLTensor                            *key,
                   const ICLTensor                            *value,
                   ICLTensor                                  *output,
                   const ScaleDotProductionAttentionLayerInfo &info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLScaleDotProductionAttentionLayer
     *
     * @param[in] output Destination tensor info. Data type supported: same as @p input
     *
     * @return a status
     */
    static Status validate(ICLTensor *output);

    // Inherited methods overridden:
    void run() override;

    private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute

#endif /* ARM_COMPUTE_CLSCALEDOTPRODUCTIONATTENTIONLAYER_H */