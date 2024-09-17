#ifndef ARM_COMPUTE_CLPOSITIONEMBEDDINGLAYER_H
#define ARM_COMPUTE_CLPOSITIONEMBEDDINGLAYER_H

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

class CLPositionEmbeddingLayer : public IFunction
{
    public:
    /** Default Constructor */
    CLPositionEmbeddingLayer();
    /** Default Destructor */
    ~CLPositionEmbeddingLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLPositionEmbeddingLayer(const CLPositionEmbeddingLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLPositionEmbeddingLayer &operator=(const CLPositionEmbeddingLayer &) = delete;

    /** Set the input and output tensor.
     * 
     * @param[in]  input        Input tensor of char text, Data type supported: U8
     * @param[in]  Position     Const tenser of Position vector, Data type supported: F32
     * @param[out] output       Output tensor, shape (seq_len,d_model). Data type supported: F32
     */
    void configure(ICLTensor *input, ICLTensor *Position, ICLTensor *output);
    /** Set the input and output tensor.
     * 
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input        Input tensor of char text, Data type supported: U8
     * @param[in]  Position     Const tenser of Position vector, Data type supported: F32
     * @param[out] output       Output tensor, shape (seq_len,d_model). Data type supported: F32
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *Position, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLPositionEmbeddingLayer
     *
     * @param[in] output Destination tensor info. Data type supported: same as @p input
     *
     * @return a status
     */
    static Status validate(ICLTensor *output);

    void prepare() override;
    // Inherited methods overridden:
    void run() override;

    private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute

#endif /* ARM_COMPUTE_CLPOSITIONEMBEDDINGLAYER_H */