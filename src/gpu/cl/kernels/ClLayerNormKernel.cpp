#include "src/gpu/cl/kernels/ClLayerNormKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{

void ClLayerNormKernel::configure(const ClCompileContext &compile_context,
                                  const ITensorInfo      *input,
                                  ITensorInfo            *output,
                                  LayerNormLayerInfo      info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate(input, output, info));
    ARM_COMPUTE_UNUSED(compile_context);

    _info                 = info;
    _input                = input;
    _output               = output;
    const int layer_width = input->dimension(info.axis());

    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output, *input->clone());

    const unsigned int vec_size_x =
        adjust_vec_size(max_cl_vector_width / input->element_size(), input->dimension(0));

    int vec_size_x_leftovers = input->dimension(0) % vec_size_x;

    Window win;
    win = calculate_max_window(*input, Steps(vec_size_x));
    IClKernel::configure_internal(win);

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option(("-DDATA_TYPE=" + get_cl_type_from_data_type(input->data_type())));
    build_opts.add_option(("-DVEC_SIZE=" + support::cpp11::to_string(vec_size_x)));
    build_opts.add_option(("-DWIDTH=" + support::cpp11::to_string(layer_width)));
    build_opts.add_option(("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(vec_size_x_leftovers)));

    std::string kernel_name("layer_norm");

    // Create kernel
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());
}

Status ClLayerNormKernel::validate(const ITensorInfo *input,
                                   const ITensorInfo *output,
                                   LayerNormLayerInfo info)
{
    ARM_COMPUTE_UNUSED(input);
    ARM_COMPUTE_UNUSED(output);
    ARM_COMPUTE_UNUSED(info);
    return Status{};
}

void ClLayerNormKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    const ICLTensor *input =
        utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    ICLTensor *output = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    Window window_in{ window };
    window_in.set(Window::DimX, Window::Dimension(0, _input->dimension(0), _input->dimension(0)));
    window_in.set(Window::DimY, Window::Dimension(0, _input->dimension(1), 1));

    Window       slice = window_in.first_slice_window_3D();
    unsigned int idx   = 0;
    add_3D_tensor_argument(idx, input, slice);
    add_3D_tensor_argument(idx, output, slice);

    _kernel.setArg<cl_float>(idx++, _info.epsilon());
    _kernel.setArg<cl_float>(idx++, _info.gamma());
    _kernel.setArg<cl_float>(idx++, _info.beta());

    enqueue(queue, *this, slice);
}

} // namespace kernels
} // namespace opencl
} // namespace arm_compute
