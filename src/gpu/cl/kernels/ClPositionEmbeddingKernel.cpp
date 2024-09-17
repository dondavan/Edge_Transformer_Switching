#include "src/gpu/cl/kernels/ClPositionEmbeddingKernel.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"

#include <cmath>

namespace arm_compute
{
namespace opencl
{
namespace kernels
{

void ClPositionEmbeddingKernel::configure(const CLCompileContext &compile_context,
                                          const ITensorInfo      *src,
                                          const ITensorInfo      *pos,
                                          ITensorInfo            *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    auto               padding_info = get_padding_info({ src, dst });
    const unsigned int vector_depth = pos->tensor_shape().x();

    // Configure output tensor info.
    auto_init_if_empty(*dst, TensorInfo(*src->clone()));

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_unsigned_type_from_element_size(src->element_size()));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(vector_depth));
    _kernel = create_kernel(compile_context, "positionalemb", build_opts.options());

    // Configure kernel window
    Window win = calculate_max_window(*dst, Steps());
    ICLKernel::configure_internal(win);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClPositionEmbeddingKernel::validate(const ITensorInfo *src, const ITensorInfo *pos, const ITensorInfo *dst)
{
    ARM_COMPUTE_UNUSED(pos);
    ARM_COMPUTE_UNUSED(src);
    ARM_COMPUTE_UNUSED(dst);

    return Status{};
}

void ClPositionEmbeddingKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_UNUSED(queue, tensors, window);

    Window slice = window.first_slice_window_3D();

    auto *src    = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_0));
    auto *vector = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_1));
    auto  dst    = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    // Set srcs
    unsigned int idx = 0;
    add_3D_tensor_argument(idx, src, window);
    add_3D_tensor_argument(idx, vector, window);
    add_3D_tensor_argument(idx, dst, window);

    enqueue(queue, *this, slice, lws_hint());
}

} // namespace kernels
} // namespace opencl
} // namespace arm_compute
