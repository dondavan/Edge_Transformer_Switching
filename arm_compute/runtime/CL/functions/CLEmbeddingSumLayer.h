#ifndef ARM_COMPUTE_CLEMBEDDINGSUMLAYER_H
#define ARM_COMPUTE_CLEMBEDDINGSUMLAYER_H

#include "arm_compute/core/Types.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

class CLEmbeddingSumLayer : public IFunction
{
    public:
    /** Default Constructor */
    CLEmbeddingSumLayer();
    /** Default Destructor */
    ~CLEmbeddingSumLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLEmbeddingSumLayer(const CLEmbeddingSumLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLEmbeddingSumLayer &operator=(const CLEmbeddingSumLayer &) = delete;

    /** Set the input and output tensor.
     * 
     * @param[in]  token        Token embedding input, Data type supported: F32
     * @param[in]  segemnt      Token embedding input, Data type supported: F32
     * @param[in]  position     Token embedding input, Data type supported: F32
     * @param[out] output       Output tensor, shape (seq_len,d_model). Data type supported: F32
     */
    void configure( ICLTensor *token,
                    ICLTensor *segemnt, 
                    ICLTensor *position, 
                    ICLTensor *output, 
                    const EmbeddingLayerInfo &emb_info);
    /** Set the input and output tensor.
     * 
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  token        Token embedding input, Data type supported: F32
     * @param[in]  segemnt      Token embedding input, Data type supported: F32
     * @param[in]  position     Token embedding input, Data type supported: F32
     * @param[out] output       Output tensor, shape (seq_len,d_model). Data type supported: F32
     */
    void configure( const CLCompileContext &compile_context,
                    ICLTensor *token,
                    ICLTensor *segemnt, 
                    ICLTensor *position, 
                    ICLTensor *output, 
                    const EmbeddingLayerInfo &emb_info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLEmbeddingSumLayer
     * 
     * @param[in]  token        Token embedding input, Data type supported: F32
     * @param[in]  segemnt      Token embedding input, Data type supported: F32
     * @param[in]  position     Token embedding input, Data type supported: F32
     * @param[out] output       Output tensor, shape (seq_len,d_model). Data type supported: F32
     *
     * @return a status
     */
    static Status validate(ICLTensor *token, ICLTensor *segemnt, ICLTensor *position, ICLTensor *output, const EmbeddingLayerInfo &emb_info);

    void prepare() override;
    // Inherited methods overridden:
    void run() override;

    private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute

#endif /* ARM_COMPUTE_CLEMBEDDINGSUMLAYER_H */