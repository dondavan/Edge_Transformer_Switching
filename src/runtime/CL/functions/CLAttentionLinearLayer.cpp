#include "arm_compute/runtime/CL/functions/CLAttentionLinearLayer.h"

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

struct CLAttentionLinearLayer::Impl
{
    MemoryGroup      memory_group{};
    IWeightsManager *weights_manager{ nullptr };

    const ITensor                  *query_input{ nullptr };
    const ITensor                  *query_w{ nullptr };
    const ITensor                  *query_b{ nullptr };
    const ITensor                  *key_input{ nullptr };
    const ITensor                  *key_w{ nullptr };
    const ITensor                  *key_b{ nullptr };
    const ITensor                  *value_input{ nullptr };
    const ITensor                  *value_w{ nullptr };
    const ITensor                  *value_b{ nullptr };
    ITensor                        *query_output{ nullptr };
    ITensor                        *key_output{ nullptr };
    ITensor                        *value_output{ nullptr };

    std::unique_ptr<opencl::ClLinear> query_kernel{ nullptr };
    std::unique_ptr<opencl::ClLinear> key_kernel{ nullptr };
    std::unique_ptr<opencl::ClLinear> value_kernel{ nullptr };

    bool is_prepared{ false };
};

CLAttentionLinearLayer::CLAttentionLinearLayer(std::shared_ptr<IMemoryManager> memory_manager,
                                               IWeightsManager                *weights_manager)
    : _impl(std::make_unique<Impl>())
{
    _impl->memory_group    = MemoryGroup(std::move(memory_manager));
    _impl->weights_manager = weights_manager;
}
CLAttentionLinearLayer::~CLAttentionLinearLayer() = default;
void CLAttentionLinearLayer::configure(const ITensor *query_input, const ITensor *query_w, const ITensor *query_b,
                                       const ITensor *key_input, const ITensor *key_w, const ITensor *key_b,
                                       const ITensor *value_input, const ITensor *value_w, const ITensor *value_b,
                                       ITensor *query_output, ITensor *key_output, ITensor *value_output,
                                       const LinearLayerInfo &linear_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), query_input, query_w, query_b,
              key_input, key_w, key_b,
              value_input, value_w, value_b,
              query_output, key_output, value_output, linear_info);
}
void CLAttentionLinearLayer::configure(const CLCompileContext &compile_context,
                                       const ITensor *query_input, const ITensor *query_w, const ITensor *query_b,
                                       const ITensor *key_input, const ITensor *key_w, const ITensor *key_b,
                                       const ITensor *value_input, const ITensor *value_w, const ITensor *value_b,
                                       ITensor *query_output, ITensor *key_output, ITensor *value_output,
                                       const LinearLayerInfo &linear_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_UNUSED(linear_info);

/*
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

#ifdef MEASURE_TIME
    auto          end_time  = std::chrono::high_resolution_clock::now();
    double        cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt", std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "CLAttentionLinearLayer::configure cost: " << cost_time << std::endl;
    measure_out.close();
#endif
*/
    _impl->query_input  = query_input;
    _impl->query_w      = query_w;
    _impl->query_b      = query_b;
    _impl->query_output = query_output;

    _impl->query_kernel = std::make_unique<opencl::ClLinear>();
    _impl->query_kernel->configure(compile_context, query_input->info(), query_w->info(), query_b->info(), query_output->info(), 1.0f, 0.f);

    _impl->key_input  = key_input;
    _impl->key_w      = key_w;
    _impl->key_b      = key_b;
    _impl->key_output = key_output;

    _impl->key_kernel = std::make_unique<opencl::ClLinear>();
    _impl->key_kernel->configure(compile_context, key_input->info(), key_w->info(), key_b->info(), key_output->info(), 1.0f, 0.f);

    _impl->value_input  = value_input;
    _impl->value_w      = value_w;
    _impl->value_b      = value_b;
    _impl->value_output = value_output;

    _impl->value_kernel = std::make_unique<opencl::ClLinear>();
    _impl->value_kernel->configure(compile_context, value_input->info(), value_w->info(), value_b->info(), value_output->info(), 1.0f, 0.f);
}

Status CLAttentionLinearLayer::validate(const ITensor *input,
                                        const ITensor *weight,
                                        const ITensor *bias, ITensor *output, const LinearLayerInfo &linear_info)
{
    ARM_COMPUTE_UNUSED(linear_info);
    return opencl::ClLinear::validate(input->info(), weight->info(), bias->info(), output->info(), 1.0f, 1.0f);
}

void CLAttentionLinearLayer::run()
{
    /*
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

#ifdef MEASURE_TIME
    auto          end_time  = std::chrono::high_resolution_clock::now();
    double        cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt", std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "CLAttentionLinearLayer::run cost: " << cost_time << std::endl;
    measure_out.close();
#endif
    */
   std::cout << "CLAttentionLinearLayer::run "<<*reinterpret_cast<float *>(_impl->query_input->ptr_to_element(Coordinates(0,0,0)));
   std::cout << "CLAttentionLinearLayer::run "<<*reinterpret_cast<float *>(_impl->key_input->ptr_to_element(Coordinates(0,0,0)));
   std::cout << "CLAttentionLinearLayer::run "<<*reinterpret_cast<float *>(_impl->value_input->ptr_to_element(Coordinates(0,0,0)));
    // Q
    ITensorPack query_pack;
    query_pack.add_tensor(TensorType::ACL_SRC_0, _impl->query_input);
    query_pack.add_tensor(TensorType::ACL_SRC_1, _impl->query_w);
    query_pack.add_tensor(TensorType::ACL_SRC_2, _impl->query_b);
    query_pack.add_tensor(TensorType::ACL_DST, _impl->query_output);
    _impl->query_kernel->run(query_pack);

    // K
    ITensorPack key_pack;
    key_pack.add_tensor(TensorType::ACL_SRC_0, _impl->key_input);
    key_pack.add_tensor(TensorType::ACL_SRC_1, _impl->key_w);
    key_pack.add_tensor(TensorType::ACL_SRC_2, _impl->key_b);
    key_pack.add_tensor(TensorType::ACL_DST, _impl->key_output);
    _impl->key_kernel->run(key_pack);

    // V
    ITensorPack value_pack;
    value_pack.add_tensor(TensorType::ACL_SRC_0, _impl->value_input);
    value_pack.add_tensor(TensorType::ACL_SRC_1, _impl->value_w);
    value_pack.add_tensor(TensorType::ACL_SRC_2, _impl->value_b);
    value_pack.add_tensor(TensorType::ACL_DST, _impl->value_output);
    _impl->value_kernel->run(value_pack);
}

} // namespace arm_compute
