#include "src/gpu/cl/operators/ClScaleDotProduction.h"

#include "arm_compute/runtime/CL/CLScheduler.h"

#include "src/common/IOperator.h"
#include "src/common/utils/LegacySupport.h"
#include "src/common/utils/Log.h"

#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/MemoryHelpers.h"

#include "src/gpu/cl/kernels/ClVectorizeKernel.h"
#include "src/gpu/cl/utils/ClAuxTensorHandler.h"

#include "src/runtime/heuristics/matmul_native/ClMatMulNativeKernelConfig.h"
#include "src/runtime/heuristics/matmul_native/IClMatMulNativeKernelConfig.h"


#ifdef MEASURE_TIME
#include <chrono>
#include <fstream>
#endif

namespace arm_compute
{
namespace opencl
{

void ClScaleDotProduction::configure(const ClCompileContext                     &compile_context,
                                     const ITensorInfo                          *query,
                                     const ITensorInfo                          *key,
                                     const ITensorInfo                          *value,
                                     ITensorInfo                                *output,
                                     const ScaleDotProductionAttentionLayerInfo &info)
{
    ARM_COMPUTE_LOG_PARAMS(key, value, query, output);
    ARM_COMPUTE_UNUSED(compile_context, query, key, value, output, info);

    // Query multi-Head reshape
    TensorShape query_reshape = TensorShape(query->tensor_shape().x() / info.h(),
                                            info.h(),
                                            query->tensor_shape().y(),
                                            1);
    _reshaped_query           = query->clone()->set_tensor_shape(query_reshape);
    TensorShape query_permute = TensorShape(query->tensor_shape().x() / info.h(),
                                            query->tensor_shape().y(),
                                            info.h(),
                                            1);
    _permuted_query           = query->clone()->set_tensor_shape(query_permute);

    auto query_reshape_kernel = std::make_unique<kernels::ClReshapeKernel>();
    query_reshape_kernel->configure(compile_context, query, &_reshaped_query);
    _query_reshape_kernel = std::move(query_reshape_kernel);

    auto query_permute_kernel = std::make_unique<kernels::ClPermuteKernel>();
    query_permute_kernel->configure(compile_context, &_reshaped_query, &_permuted_query, PermutationVector(0U, 2U, 1U));
    _query_permute_kernel = std::move(query_permute_kernel);

    // Key multi-Head reshape
    TensorShape key_reshape = TensorShape(key->tensor_shape().x() / info.h(),
                                          info.h(),
                                          key->tensor_shape().y(),
                                          1);
    _reshaped_key           = key->clone()->set_tensor_shape(key_reshape);
    TensorShape key_permute = TensorShape(key->tensor_shape().x() / info.h(),
                                          key->tensor_shape().y(),
                                          info.h(),
                                          1);
    _permuted_key           = key->clone()->set_tensor_shape(key_permute);

    auto key_reshape_kernel = std::make_unique<kernels::ClReshapeKernel>();
    key_reshape_kernel->configure(compile_context, key, &_reshaped_key);
    _key_reshape_kernel = std::move(key_reshape_kernel);

    auto key_permute_kernel = std::make_unique<kernels::ClPermuteKernel>();
    key_permute_kernel->configure(compile_context, &_reshaped_key, &_permuted_key, PermutationVector(0U, 2U, 1U));
    _key_permute_kernel = std::move(key_permute_kernel);

    // Value multi-Head reshape
    TensorShape value_reshape = TensorShape(value->tensor_shape().x() / info.h(),
                                            info.h(),
                                            value->tensor_shape().y(),
                                            1);
    _reshaped_value           = value->clone()->set_tensor_shape(value_reshape);
    TensorShape value_permute = TensorShape(value->tensor_shape().x() / info.h(),
                                            value->tensor_shape().y(),
                                            info.h(),
                                            1);
    _permuted_value           = value->clone()->set_tensor_shape(value_permute);

    auto value_reshape_kernel = std::make_unique<kernels::ClReshapeKernel>();
    value_reshape_kernel->configure(compile_context, value, &_reshaped_value);
    _value_reshape_kernel = std::move(value_reshape_kernel);

    auto value_permute_kernel = std::make_unique<kernels::ClPermuteKernel>();
    value_permute_kernel->configure(compile_context, &_reshaped_value, &_permuted_value, PermutationVector(0U, 2U, 1U));
    _value_permute_kernel = std::move(value_permute_kernel);

    // Specify whether transpose weights is necessary in matmul info
    const MatMulInfo mat_info_qk = MatMulInfo().adj_rhs(true);

    // Note: MatMul does not need offset negation unlike gemm
    // 1. Change shape when calling matmul to fit batch expectations.
    //_lhs_to_use = src->clone()->set_tensor_shape(get_reshaped_matmul_tensor(_lhs_to_use.tensor_shape()));

    // 2. Use heuristics to get kernel info object
    const GPUTarget                                         gpu_target = CLScheduler::get().target();
    std::unique_ptr<cl_matmul::IClMatMulNativeKernelConfig> kernel_config_qk =
        cl_matmul::ClMatMulNativeKernelConfigurationFactory::create(gpu_target);
    MatMulKernelInfo mm_kernel_info_qk = kernel_config_qk->configure(&_permuted_query, &_permuted_key, mat_info_qk);

    // Matrix multiply compute multi-head attention between Query and Key
    auto        product_mm_kernel = std::make_unique<kernels::ClLinearKernel>();
    const float scale             = 1.0f / sqrt(info.d_model() / info.h());
    product_mm_kernel->set_target(gpu_target);
    product_mm_kernel->configure(compile_context, &_permuted_query, &_permuted_key, nullptr, &_scaled_query_key, scale, 0, mm_kernel_info_qk);
    _product_mm_kernel = std::move(product_mm_kernel);

    //  Softmax of previous product
    SoftmaxKernelInfo softmax_info{ 1.0f, false, query->data_type(), 0 };
    auto              softmax_kernel = std::make_unique<kernels::ClSoftmaxKernel>();
    softmax_kernel->configure(compile_context, _scaled_query_key, _softmaxed_product, softmax_info);
    _softmax_kernel = std::move(softmax_kernel);

    // Specify whether transpose weights is necessary in matmul info
    const MatMulInfo mat_info_pv = MatMulInfo();

    // Note: MatMul does not need offset negation unlike gemm
    // 1. Change shape when calling matmul to fit batch expectations.
    //_lhs_to_use = src->clone()->set_tensor_shape(get_reshaped_matmul_tensor(_lhs_to_use.tensor_shape()));

    // 2. Use heuristics to get kernel info object
    std::unique_ptr<cl_matmul::IClMatMulNativeKernelConfig> kernel_config_pv =
        cl_matmul::ClMatMulNativeKernelConfigurationFactory::create(gpu_target);
    MatMulKernelInfo mm_kernel_info_pv = kernel_config_pv->configure(&_softmaxed_product, &_permuted_value, mat_info_pv);

    //  Multiply between scaled product and value
    auto context_mm_kernel = std::make_unique<kernels::ClLinearKernel>();
    context_mm_kernel->set_target(gpu_target);
    context_mm_kernel->configure(compile_context, &_softmaxed_product, &_permuted_value, nullptr, &_gemmed_context, 1.0f, 0, mm_kernel_info_pv);
    _context_mm_kernel = std::move(context_mm_kernel);

    // Concat multi-Head reshape
    TensorShape concat_permute = TensorShape(query->tensor_shape().x() / info.h(),
                                             info.h(),
                                             query->tensor_shape().y(),
                                             1);
    _permuted_concat           = query->clone()->set_tensor_shape(concat_permute);

    auto concat_permute_kernel = std::make_unique<kernels::ClPermuteKernel>();
    concat_permute_kernel->configure(compile_context, &_gemmed_context, &_permuted_concat, PermutationVector(0U, 2U, 1U));
    _concat_permute_kernel = std::move(concat_permute_kernel);

    auto concat_reshape_kernel = std::make_unique<kernels::ClReshapeKernel>();
    concat_reshape_kernel->configure(compile_context, &_permuted_concat, output);
    _concat_reshape_kernel = std::move(concat_reshape_kernel);
}

Status
ClScaleDotProduction::validate(const ITensorInfo *query, const ITensorInfo *key, const ITensorInfo *value, ITensorInfo *output)
{
    ARM_COMPUTE_UNUSED(key);
    ARM_COMPUTE_UNUSED(value);
    ARM_COMPUTE_UNUSED(query);
    ARM_COMPUTE_UNUSED(output);
    return Status{};
}

void ClScaleDotProduction::run(ITensorPack &tensors)
{
    ARM_COMPUTE_UNUSED(tensors);

    auto query  = tensors.get_const_tensor(ACL_SRC_0);
    auto key    = tensors.get_const_tensor(ACL_SRC_1);
    auto value  = tensors.get_const_tensor(ACL_SRC_2);
    auto output = tensors.get_tensor(ACL_DST);

    CLAuxTensorHandler reshaped_query(offset_int_vec(QueryReshape), _reshaped_query, tensors);
    CLAuxTensorHandler permuted_query(offset_int_vec(QueryPermute), _permuted_query, tensors);
    CLAuxTensorHandler reshaped_key(offset_int_vec(KeyReshape), _reshaped_key, tensors);
    CLAuxTensorHandler permuted_key(offset_int_vec(KeyPermute), _permuted_key, tensors);
    CLAuxTensorHandler reshaped_value(offset_int_vec(ValueReshape), _reshaped_value, tensors);
    CLAuxTensorHandler permuted_value(offset_int_vec(ValuePermute), _permuted_value, tensors);
    CLAuxTensorHandler scaled_query_key(offset_int_vec(QueryKeyScale), _scaled_query_key, tensors);
    CLAuxTensorHandler softmaxed_product(offset_int_vec(Softmax), _softmaxed_product, tensors);
    CLAuxTensorHandler gemmed_context(offset_int_vec(GemmedContext), _gemmed_context, tensors);
    CLAuxTensorHandler permuted_concat(offset_int_vec(ConcatPermute), _permuted_concat, tensors);

    // Run Query multi-Head reshape
    ITensorPack query_reshape_pack{ { ACL_SRC_0, query }, { ACL_DST, reshaped_query.get() } };
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif
    CLScheduler::get().enqueue_op(*_query_reshape_kernel, query_reshape_pack, true);
#ifdef MEASURE_TIME
    auto   end_time  = std::chrono::high_resolution_clock::now();
    double cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt",std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "query_reshape cost: " << cost_time << std::endl;
#endif
    ITensorPack query_permute_pack{ { ACL_SRC, reshaped_query.get() }, { ACL_DST, permuted_query.get() } };
#ifdef MEASURE_TIME
    start_time = std::chrono::high_resolution_clock::now();
#endif
    CLScheduler::get().enqueue_op(*_query_permute_kernel, query_permute_pack, true);
#ifdef MEASURE_TIME
    end_time  = std::chrono::high_resolution_clock::now();
    cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    measure_out.precision(5);
    measure_out << std::scientific << "query_permute_func cost: " << cost_time << std::endl;
#endif

    // Run Key multi-Head reshape
    ITensorPack key_reshape_pack{ { ACL_SRC_0, key }, { ACL_DST, reshaped_key.get() } };
#ifdef MEASURE_TIME
    start_time = std::chrono::high_resolution_clock::now();
#endif
    CLScheduler::get().enqueue_op(*_key_reshape_kernel, key_reshape_pack, true);
#ifdef MEASURE_TIME
    end_time  = std::chrono::high_resolution_clock::now();
    cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    measure_out.precision(5);
    measure_out << std::scientific << "key_reshape cost: " << cost_time << std::endl;
#endif
    ITensorPack key_permute_pack{ { ACL_SRC, reshaped_key.get() }, { ACL_DST, permuted_key.get() } };
#ifdef MEASURE_TIME
    start_time = std::chrono::high_resolution_clock::now();
#endif
    CLScheduler::get().enqueue_op(*_key_permute_kernel, key_permute_pack, true);
#ifdef MEASURE_TIME
    end_time  = std::chrono::high_resolution_clock::now();
    cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    measure_out.precision(5);
    measure_out << std::scientific << "key_permute_func cost: " << cost_time << std::endl;
#endif

    // Run Value multi-Head reshape
    ITensorPack value_reshape_pack{ { ACL_SRC_0, value }, { ACL_DST, reshaped_value.get() } };
#ifdef MEASURE_TIME
    start_time = std::chrono::high_resolution_clock::now();
#endif
    CLScheduler::get().enqueue_op(*_value_reshape_kernel, value_reshape_pack, true);
#ifdef MEASURE_TIME
    end_time  = std::chrono::high_resolution_clock::now();
    cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    measure_out.precision(5);
    measure_out << std::scientific << "value_reshape cost: " << cost_time << std::endl;
#endif
    ITensorPack value_permute_pack{ { ACL_SRC, reshaped_value.get() }, { ACL_DST, permuted_value.get() } };
#ifdef MEASURE_TIME
    start_time = std::chrono::high_resolution_clock::now();
#endif
    CLScheduler::get().enqueue_op(*_value_permute_kernel, value_permute_pack, true);
#ifdef MEASURE_TIME
    end_time  = std::chrono::high_resolution_clock::now();
    cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    measure_out.precision(5);
    measure_out << std::scientific << "value_permute_func cost: " << cost_time << std::endl;
#endif


    // Run matrix multiply compute multi-head attention between Query and Key
    ITensorPack gemm_QK_pack{ { ACL_SRC_0, permuted_query.get() }, { ACL_SRC_1, permuted_key.get() }, { ACL_DST, scaled_query_key.get() } };
#ifdef MEASURE_TIME
    start_time = std::chrono::high_resolution_clock::now();
#endif
    CLScheduler::get().enqueue_op(*_product_mm_kernel, gemm_QK_pack, true);
#ifdef MEASURE_TIME
    end_time  = std::chrono::high_resolution_clock::now();
    cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    measure_out.precision(5);
    measure_out << std::scientific << "MMUL QK cost: " << cost_time << std::endl;
#endif

    // Softmax scaled product
    ITensorPack softmax_pack = { { ACL_SRC, scaled_query_key.get() }, { ACL_DST, softmaxed_product.get() } };
#ifdef MEASURE_TIME
    start_time = std::chrono::high_resolution_clock::now();
#endif
    CLScheduler::get().enqueue_op(*_softmax_kernel, softmax_pack, true);
#ifdef MEASURE_TIME
    end_time  = std::chrono::high_resolution_clock::now();
    cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    measure_out.precision(5);
    measure_out << std::scientific << "softmax cost: " << cost_time << std::endl;
#endif

    // Run matrix multiply compute multi-head attention between Context and Value
    ITensorPack gemm_context_pack{ { ACL_SRC_0, softmaxed_product.get() }, { ACL_SRC_1, permuted_value.get() }, { ACL_DST, gemmed_context.get() } };
#ifdef MEASURE_TIME
    start_time = std::chrono::high_resolution_clock::now();
#endif
    CLScheduler::get().enqueue_op(*_context_mm_kernel, gemm_context_pack, true);
#ifdef MEASURE_TIME
    end_time  = std::chrono::high_resolution_clock::now();
    cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    measure_out.precision(5);
    measure_out << std::scientific << "MMUL CV cost: " << cost_time << std::endl;
#endif

    // Concat all attention head together
    ITensorPack concat_permute_pack{ { ACL_SRC, gemmed_context.get() }, { ACL_DST, permuted_concat.get() } };
#ifdef MEASURE_TIME
    start_time = std::chrono::high_resolution_clock::now();
#endif
    CLScheduler::get().enqueue_op(*_concat_permute_kernel, concat_permute_pack, true);
#ifdef MEASURE_TIME
    end_time  = std::chrono::high_resolution_clock::now();
    cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    measure_out.precision(5);
    measure_out << std::scientific << "concat_permute_func cost: " << cost_time << std::endl;
#endif

    ITensorPack concat_reshape_pack{ { ACL_SRC_0, permuted_concat.get() }, { ACL_DST, output } };
#ifdef MEASURE_TIME
    start_time = std::chrono::high_resolution_clock::now();
#endif
    CLScheduler::get().enqueue_op(*_concat_reshape_kernel, concat_reshape_pack, true);
#ifdef MEASURE_TIME
    end_time  = std::chrono::high_resolution_clock::now();
    cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    measure_out.precision(5);
    measure_out << std::scientific << "concat_reshape cost: " << cost_time << std::endl;
#endif
}

experimental::MemoryRequirements ClScaleDotProduction::workspace() const
{
    return _aux_mem;
}

} // namespace opencl
} // namespace arm_compute
