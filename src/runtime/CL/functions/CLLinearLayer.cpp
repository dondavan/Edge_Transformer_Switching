#include "arm_compute/runtime/CL/functions/CLLinearLayer.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Validate.h"

#include "src/core/CL/ICLKernel.h"
#include "src/core/helpers/MemoryHelpers.h"

#include "src/gpu/cl/operators/ClLinear.h"

#ifdef MEASURE_TIME
#include <chrono>
#include <fstream>
#endif

namespace arm_compute
{

struct CLLinearLayer::Impl
{
    MemoryGroup      memory_group{};
    IWeightsManager *weights_manager{ nullptr };

    const ICLTensor                  *src{ nullptr };
    const ICLTensor                  *weight{ nullptr };
    const ICLTensor                  *bias{ nullptr };
    ICLTensor                        *dst{ nullptr };
    std::unique_ptr<opencl::ClLinear> op{ nullptr };

    bool is_prepared{false};
};

CLLinearLayer::CLLinearLayer(std::shared_ptr<IMemoryManager> memory_manager,
                             IWeightsManager                *weights_manager)
    : _impl(std::make_unique<Impl>())
{
    _impl->memory_group    = MemoryGroup(std::move(memory_manager));
    _impl->weights_manager = weights_manager;
}
CLLinearLayer::~CLLinearLayer() = default;
void CLLinearLayer::configure(const ICLTensor *input,
                              const ICLTensor *weight,
                              const ICLTensor *bias, ICLTensor *output, const LinearLayerInfo &linear_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, weight, bias, output, linear_info);
}
void CLLinearLayer::configure(const CLCompileContext &compile_context,
                              const ICLTensor        *input,
                              const ICLTensor        *weight,
                              const ICLTensor *bias, ICLTensor *output, const LinearLayerInfo &linear_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_UNUSED(linear_info);

#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif


    std::cout << "CLLinearLayer::configure start" << std::endl;
    _impl->src    = input;
    _impl->weight = weight;
    _impl->bias   = bias;
    _impl->dst    = output;

    _impl->op = std::make_unique<opencl::ClLinear>();
    _impl->op->configure(compile_context, input->info(), weight->info(), bias->info(), output->info(), 1.0f, 0.f);

    std::cout << "CLLinearLayer::configure end" << std::endl;

#ifdef MEASURE_TIME
    auto          end_time  = std::chrono::high_resolution_clock::now();
    double        cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt", std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "CLLinearLayer::configure cost: " << cost_time << std::endl;
    measure_out.close();
#endif
}

Status CLLinearLayer::validate(const ICLTensor *input,
                               const ICLTensor *weight,
                               const ICLTensor *bias, ICLTensor *output, const LinearLayerInfo &linear_info)
{
    ARM_COMPUTE_UNUSED(linear_info);
    return opencl::ClLinear::validate(input->info(), weight->info(), bias->info(), output->info(), 1.0f, 1.0f);
}


void CLLinearLayer::run()
{
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

    ITensorPack pack;

    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->weight);
    pack.add_tensor(TensorType::ACL_SRC_2, _impl->bias);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);

    _impl->op->run(pack);

#ifdef MEASURE_TIME
    auto          end_time  = std::chrono::high_resolution_clock::now();
    double        cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt", std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "CLLinearLayer::run cost: " << cost_time << std::endl;
    measure_out.close();
#endif
}

} // namespace arm_compute
