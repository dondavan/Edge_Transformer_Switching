#ifndef ARM_COMPUTE_CLLINEAR_LAYER_H
#define ARM_COMPUTE_CLLINEAR_LAYER_H

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
class ICLTensor;
class ICLTensorInfo;

/** Perform basic linear function */
class CLLinearLayer : public IFunction
{
    public:
    /** Constructor */
    CLLinearLayer(std::shared_ptr<IMemoryManager> memory_manager  = nullptr,
                  IWeightsManager                *weights_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLLinearLayer(const CLLinearLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    CLLinearLayer(CLLinearLayer &&) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLLinearLayer &operator=(const CLLinearLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    CLLinearLayer &operator=(CLLinearLayer &&) = delete;
    /** Destructor */
    ~CLLinearLayer();

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
    void configure(const ICLTensor       *input1,
                   const ICLTensor       *weight,
                   const ICLTensor       *bias,
                   ICLTensor             *output,
                   const LinearLayerInfo &linear_info);
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
                   const ICLTensor        *input1,
                   const ICLTensor        *weight,
                   const ICLTensor        *bias,
                   ICLTensor              *output,
                   const LinearLayerInfo  &linear_info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLLinearLayer
     *
     * @param[in] input1 First input tensor info. Data types supported: F32.
     * @param[in] output Output tensor info. Data type supported: F32.
     *
     * @return a status
     */
    static Status validate(const ICLTensor *input, const ICLTensor *weight, const ICLTensor *bias, ICLTensor *output, const LinearLayerInfo &linear_info);

    // Inherited methods overridden
    void run() override;

    private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace arm_compute

#endif /* ARM_COMPUTE_CLLINEAR_LAYER_H */