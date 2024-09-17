#ifndef ARM_COMPUTE_CLLAYER_NORM_LAYER_H
#define ARM_COMPUTE_CLLAYER_NORM_LAYER_H

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/CL/ICLOperator.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class CLCompileContext;
class ICLTensor;
class ICLTensorInfo;

/** Perform basic layer normalization function */
class CLLayerNormLayer : public IFunction
{
    public:
    /** Constructor */
    CLLayerNormLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLLayerNormLayer(const CLLayerNormLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    CLLayerNormLayer(CLLayerNormLayer &&) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLLayerNormLayer &operator=(const CLLayerNormLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    CLLayerNormLayer &operator=(CLLayerNormLayer &&) = delete;
    /** Destructor */
    ~CLLayerNormLayer();

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
    void configure(const ICLTensor          *input,
                   ICLTensor                *output,
                   const LayerNormLayerInfo &LayerNorm_info);
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
    void configure(const CLCompileContext   &compile_context,
                   const ICLTensor          *input,
                   ICLTensor                *output,
                   const LayerNormLayerInfo &LayerNorm_info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLLayerNormLayer
     *
     * @param[in] input     First input tensor info. Data types supported: F32.
     * @param[in] output Output tensor info. Data type supported: F32.
     *
     * @return a status
     */
    static Status validate(const ICLTensor *input, ICLTensor *output, const LayerNormLayerInfo &LayerNorm_info);

    // Inherited methods overridden
    void run() override;

    private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace arm_compute

#endif /* ARM_COMPUTE_CLLAYER_NORM_LAYER_H */