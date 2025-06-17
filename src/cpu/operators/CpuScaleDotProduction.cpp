#include "src/cpu/operators/CpuScaleDotProduction.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/common/IOperator.h"
#include "src/common/utils/LegacySupport.h"
#include "src/common/utils/Log.h"
#include "src/cpu/CpuContext.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/utils/CpuAuxTensorHandler.h"

#include "arm_compute/runtime/CL/CLTensor.h"

#ifdef MEASURE_TIME
#include <chrono>
#include <fstream>
#endif


namespace arm_compute
{
std::unique_ptr<ITensor> create_mask(TensorInfo * mask_target)
{
    // Create tensor info
    TensorInfo mask_info = *mask_target->clone();
    auto mask_tensor = std::make_unique<Tensor>();
    mask_tensor->allocator()->init(mask_info);
    mask_tensor->allocator()->allocate();

    // Fill mask manually
    Window window;
    window.use_tensor_dimensions(mask_info.tensor_shape());

    Iterator it(mask_tensor.get(), window);
    execute_window_loop(window, [&](const Coordinates &coords) {
        const int i = coords.y();
        const int j = coords.x();

        float *ptr = reinterpret_cast<float *>(it.ptr());
        *ptr = (j > i) ? -1e9f : 0.0f;
    }, it);

    return mask_tensor;
}
namespace cpu
{

void CpuScaleDotProduction::configure(const ITensorInfo                 *query,
                                      const ITensorInfo                 *key,
                                      const ITensorInfo                 *value,
                                      ITensorInfo                       *output,
                                      const ScaleDotProductionLayerInfo &info,
                                      int                                recurrence_count)
{
    _recurrence_count = recurrence_count;

    ARM_COMPUTE_LOG_PARAMS(key, value, query, output);

    TensorShape query_buffer = TensorShape(query->tensor_shape().x(),
                                           query->tensor_shape().y(),
                                           query->tensor_shape().z(),
                                           1);
    TensorShape key_buffer   = TensorShape(key->tensor_shape().x(),
                                           key->tensor_shape().y(),
                                           key->tensor_shape().z(),
                                           1);
    TensorShape value_buffer = TensorShape(value->tensor_shape().x(),
                                           value->tensor_shape().y(),
                                           value->tensor_shape().z(),
                                           1);
    TensorShape output_buffer = TensorShape(output->tensor_shape().x(),
                                           output->tensor_shape().y(),
                                           output->tensor_shape().z(),
                                           1);
    _query_cpu_buffer        = query->clone()->set_tensor_shape(query_buffer);
    _key_cpu_buffer          = key->clone()->set_tensor_shape(key_buffer);
    _value_cpu_buffer        = value->clone()->set_tensor_shape(value_buffer);
    _ouput_cpu_buffer        = output->clone()->set_tensor_shape(output_buffer);

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
    _query_reshape_kernel     = std::make_unique<kernels::CpuReshapeKernel>();
    _query_reshape_kernel->configure(query, &_reshaped_query);
    _query_permute_func = std::make_unique<CpuPermute>();
    _query_permute_func->configure(&_reshaped_query, &_permuted_query, PermutationVector(0U, 2U, 1U));

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
    _key_reshape_kernel     = std::make_unique<kernels::CpuReshapeKernel>();
    _key_reshape_kernel->configure(key, &_reshaped_key);
    _key_permute_func = std::make_unique<CpuPermute>();
    _key_permute_func->configure(&_reshaped_key, &_permuted_key, PermutationVector(0U, 2U, 1U));
    // Pretranspose Key, K=K^T
    _key_transpose_func = std::make_unique<CpuTranspose>();
    _key_transpose_func->configure(&_permuted_key, &_transposed_key);

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
    _value_reshape_kernel     = std::make_unique<kernels::CpuReshapeKernel>();
    _value_reshape_kernel->configure(value, &_reshaped_value);
    _value_permute_func = std::make_unique<CpuPermute>();
    _value_permute_func->configure(&_reshaped_value, &_permuted_value, PermutationVector(0U, 2U, 1U));

    // Configure interleave kernel
    _query_interleave_kernel = std::make_unique<cpu::kernels::CpuGemmInterleave4x4Kernel>();
    _query_interleave_kernel->configure(&_permuted_query, &_tmp_query);
    _aux_mem[InterleavedLHS] =
        experimental::MemoryInfo(offset_int_vec(InterleavedLHS), experimental::MemoryLifetime::Persistent, _tmp_query.total_size());

    // Configure rhs transpose1xw kernel
    _key_transpose1xW_kernel = std::make_unique<cpu::kernels::CpuGemmTranspose1xWKernel>();
    _key_transpose1xW_kernel->configure(&_transposed_key, &_tmp_key);
    _aux_mem[Transposed1xWRHS] =
        experimental::MemoryInfo(offset_int_vec(Transposed1xWRHS), experimental::MemoryLifetime::Persistent, _tmp_key.total_size());

    // Matrix multiply compute multi-head attention between Query and Key
    _product_mm_kernel = std::make_unique<cpu::kernels::CpuGemmMatrixMultiplyKernel>();
    const int   m      = _permuted_query.dimension(1);
    const int   n      = _transposed_key.dimension(0);
    const int   k      = _permuted_query.dimension(0);
    const float scale  = 1.0f / sqrt(info.d_model() / info.h());
    _product_mm_kernel->configure(&_tmp_query, &_tmp_key, &_scaled_query_key, scale, true, GEMMReshapeInfo(m, n, k));

    // enable and configure masking of the query-key product
    _is_masked = info.is_masked();
    if (_is_masked)
    {
        _masked_scaled_kq = *_scaled_query_key.clone();
        _mask_info = *_scaled_query_key.clone();
        _masking_kernel = std::make_unique<kernels::CpuAddKernel>();
        _masking_kernel->configure(&_scaled_query_key, &_mask_info, &_masked_scaled_kq, ConvertPolicy::WRAP);
    }

    //  Softmax of previous product 
    _softmax_func = std::make_unique<cpu::CpuSoftmaxGeneric>();
    _softmax_func->configure(&_scaled_query_key, &_softmaxed_product);

    // Configure interleave kernel
    _product_interleave_kernel = std::make_unique<cpu::kernels::CpuGemmInterleave4x4Kernel>();
    _product_interleave_kernel->configure(&_softmaxed_product, &_interleaved_product);
    _aux_mem[InterleavedProduct] =
        experimental::MemoryInfo(offset_int_vec(InterleavedProduct), experimental::MemoryLifetime::Persistent, _interleaved_product.total_size());

    // Configure rhs transpose1xw kernel
    _value_transpose1xW_kernel = std::make_unique<cpu::kernels::CpuGemmTranspose1xWKernel>();
    _value_transpose1xW_kernel->configure(&_permuted_value, &_transposed1xW_value);
    _aux_mem[Transposed1xWValue] =
        experimental::MemoryInfo(offset_int_vec(Transposed1xWValue), experimental::MemoryLifetime::Persistent, _transposed1xW_value.total_size());

    //  Multiply between scaled product and value
    _context_mm_kernel = std::make_unique<cpu::kernels::CpuGemmMatrixMultiplyKernel>();
    const int m1       = _softmaxed_product.dimension(1);
    const int n1       = _permuted_value.dimension(0);
    const int k1       = _softmaxed_product.dimension(0);
    _context_mm_kernel->configure(&_interleaved_product, &_permuted_value, &_gemmed_context, 1.0f, true, GEMMReshapeInfo(m1, n1, k1));

    // Concat multi-Head reshape

    TensorShape concat_permute = TensorShape(query->tensor_shape().x() / info.h(),
                                             info.h(),
                                             query->tensor_shape().y(),
                                             1);
    _permuted_concat           = query->clone()->set_tensor_shape(concat_permute);
    _concat_permute_func       = std::make_unique<CpuPermute>();
    _concat_permute_func->configure(&_gemmed_context, &_permuted_concat, PermutationVector(0U, 2U, 1U));

    _concat_reshape_kernel = std::make_unique<kernels::CpuReshapeKernel>();
    _concat_reshape_kernel->configure(&_permuted_concat, output);
}

Status
CpuScaleDotProduction::validate(const ITensorInfo *query, const ITensorInfo *key, const ITensorInfo *value, ITensorInfo *output)
{
    ARM_COMPUTE_UNUSED(key);
    ARM_COMPUTE_UNUSED(value);
    ARM_COMPUTE_UNUSED(query);
    ARM_COMPUTE_UNUSED(output);
    return Status{};
}

void CpuScaleDotProduction::run(ITensorPack &tensors)
{
    ARM_COMPUTE_UNUSED(tensors);

    auto query  = tensors.get_tensor(ACL_SRC_0);
    auto key    = tensors.get_tensor(ACL_SRC_1);
    auto value  = tensors.get_tensor(ACL_SRC_2);
    auto output = tensors.get_tensor(ACL_DST);

    std::cout << "query id: " << query->info()->id() << std::endl;
    std::cout << "key id: " << key->info()->id() << std::endl;
    std::cout << "value id: " << value->info()->id() << std::endl;
    std::cout << "output id: " << output->info()->id() << std::endl;

    ICLTensor *query_cl;
    ICLTensor *key_cl;
    ICLTensor *value_cl;
    //ICLTensor *output_cl;

#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif
    /*
if(_recurrence_count ==0){
    if(query->info()->tensor_target_type() == TensorTargetType::CL)
    {
        ITensor *query_nc = const_cast<ITensor *>(query);
        query_cl          = static_cast<ICLTensor *>(query_nc);
        query_cl->map(CLScheduler::get().queue());
    }

    if(key->info()->tensor_target_type() == TensorTargetType::CL)
    {
        ITensor *key_nc = const_cast<ITensor *>(key);
        key_cl          = static_cast<ICLTensor *>(key_nc);
        key_cl->map(CLScheduler::get().queue());
    }

    if(value->info()->tensor_target_type() == TensorTargetType::CL)
    {
        ITensor *value_nc = const_cast<ITensor *>(value);
        value_cl          = static_cast<ICLTensor *>(value_nc);
        value_cl->map(CLScheduler::get().queue());
    }

    if(output->info()->tensor_target_type() == TensorTargetType::CL)
    {
        output_cl          = static_cast<ICLTensor *>(output);
        output_cl->map(CLScheduler::get().queue());
    }
}
    std::cout << "query_cl CL_value: " << *reinterpret_cast<float *>(query->ptr_to_element(Coordinates(0,0,0))) << std::endl;
    std::cout << "key_cl CL_value: " << *reinterpret_cast<float *>(key->ptr_to_element(Coordinates(0,0,0))) << std::endl;
    std::cout << "value_cl CL_value: " << *reinterpret_cast<float *>(value->ptr_to_element(Coordinates(0,0,0))) << std::endl;
    std::cout << "output_cl CL_value: " << *reinterpret_cast<float *>(output->ptr_to_element(Coordinates(0,0,0))) << std::endl;
    */
#ifdef MEASURE_TIME
    auto          end_time  = std::chrono::high_resolution_clock::now();
    double        cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt", std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "mapping cost: " << cost_time << std::endl;
#endif

#ifdef MEASURE_TIME
    auto read_start_time = std::chrono::high_resolution_clock::now();
#endif
    
    CpuAuxTensorHandler query_cpu_buffer_aux(offset_int_vec(QueryCPUBuffer), _query_cpu_buffer, tensors);
    CpuAuxTensorHandler key_cpu_buffer_aux(offset_int_vec(ValueCPUBuffer), _key_cpu_buffer, tensors);
    CpuAuxTensorHandler value_cpu_buffer_aux(offset_int_vec(KeyCPUBuffer), _value_cpu_buffer, tensors);
    CpuAuxTensorHandler output_cpu_buffer_aux(offset_int_vec(OuputCPUBuffer), _ouput_cpu_buffer, tensors);

    if(query->info()->tensor_target_type() == TensorTargetType::CL)
    {
        ITensor *query_nc = const_cast<ITensor *>(query);
        query_cl          = static_cast<ICLTensor *>(query_nc);
        if(_recurrence_count ==0)
        {
            query_cl->map(CLScheduler::get().queue());
        }
        CLScheduler::get().queue().enqueueReadBuffer(query_cl->cl_buffer(), CL_TRUE, 0, query_cpu_buffer_aux.get()->info()->total_size(), query_cpu_buffer_aux.get()->buffer());
        std::cout << "Aux CL_query: " << *reinterpret_cast<float *>(query_cpu_buffer_aux.get()->ptr_to_element(Coordinates(0,0,0))) << std::endl;
    }

    if(key->info()->tensor_target_type() == TensorTargetType::CL)
    {
        ITensor *key_nc = const_cast<ITensor *>(key);
        key_cl          = static_cast<ICLTensor *>(key_nc);
        if(_recurrence_count ==0)
        {
            key_cl->map(CLScheduler::get().queue());
        }
        CLScheduler::get().queue().enqueueReadBuffer(key_cl->cl_buffer(), CL_TRUE, 0, key_cpu_buffer_aux.get()->info()->total_size(), key_cpu_buffer_aux.get()->buffer());
        std::cout << "Aux CL_key: " << *reinterpret_cast<float *>(key_cpu_buffer_aux.get()->ptr_to_element(Coordinates(0,0,0))) << std::endl;
    }

    if(value->info()->tensor_target_type() == TensorTargetType::CL)
    {
        ITensor *value_nc = const_cast<ITensor *>(value);
        value_cl          = static_cast<ICLTensor *>(value_nc);
        if(_recurrence_count ==0)
        {
            value_cl->map(CLScheduler::get().queue());
        }
        CLScheduler::get().queue().enqueueReadBuffer(value_cl->cl_buffer(), CL_TRUE, 0, value_cpu_buffer_aux.get()->info()->total_size(), value_cpu_buffer_aux.get()->buffer());
        std::cout << "Aux CL_value: " << *reinterpret_cast<float *>(value_cpu_buffer_aux.get()->ptr_to_element(Coordinates(0,0,0))) << std::endl;
    }
    //CLScheduler::get().queue().enqueueReadBuffer(query_cl->cl_buffer(), CL_TRUE, 0, query_cpu_buffer_aux.get()->info()->total_size(), query_cpu_buffer_aux.get()->buffer());
    //CLScheduler::get().queue().enqueueReadBuffer(key_cl->cl_buffer(), CL_TRUE, 0, key_cpu_buffer_aux.get()->info()->total_size(), value_cpu_buffer_aux.get()->buffer());
    //CLScheduler::get().queue().enqueueReadBuffer(value_cl->cl_buffer(), CL_TRUE, 0, value_cpu_buffer_aux.get()->info()->total_size(), key_cpu_buffer_aux.get()->buffer());


#ifdef MEASURE_TIME
    auto   read_end_time  = std::chrono::high_resolution_clock::now();
    double read_cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(read_end_time - read_start_time).count();
    measure_out.precision(5);
    measure_out << std::scientific << "Reading cost: " << read_cost_time << std::endl;
#endif

    CpuAuxTensorHandler reshaped_query(offset_int_vec(QueryReshape), _reshaped_query, tensors);
    CpuAuxTensorHandler permuted_query(offset_int_vec(QueryPermute), _permuted_query, tensors);
    CpuAuxTensorHandler reshaped_key(offset_int_vec(KeyReshape), _reshaped_key, tensors);
    CpuAuxTensorHandler permuted_key(offset_int_vec(KeyPermute), _permuted_key, tensors);
    CpuAuxTensorHandler transposed_key(offset_int_vec(KeyTranspose), _transposed_key, tensors);
    CpuAuxTensorHandler reshaped_value(offset_int_vec(ValueReshape), _reshaped_value, tensors);
    CpuAuxTensorHandler permuted_value(offset_int_vec(ValuePermute), _permuted_value, tensors);
    CpuAuxTensorHandler permuted_concat(offset_int_vec(ConcatPermute), _permuted_concat, tensors);

    CpuAuxTensorHandler scaled_query_key(offset_int_vec(QueryKeyScale), _scaled_query_key, tensors);
    CpuAuxTensorHandler interleaved_query(offset_int_vec(InterleavedLHS), _tmp_query, tensors, true);
    CpuAuxTensorHandler transposed1xw_key(offset_int_vec(Transposed1xWRHS), _tmp_key, tensors, true);
    CpuAuxTensorHandler softmaxed_product(offset_int_vec(Softmax), _softmaxed_product, tensors);
    CpuAuxTensorHandler interleaved_product(offset_int_vec(InterleavedProduct), _interleaved_product, tensors, true);
    CpuAuxTensorHandler transposed1xW_value(offset_int_vec(Transposed1xWValue), _transposed1xW_value, tensors, true);
    CpuAuxTensorHandler gemmed_context(offset_int_vec(GemmedContext), _gemmed_context, tensors);

    // Run Query multi-Head reshape
    ITensorPack query_reshape_pack{ { ACL_SRC_0, query_cpu_buffer_aux.get() }, { ACL_DST, reshaped_query.get() } };
    NEScheduler::get().schedule_op(_query_reshape_kernel.get(), Window::DimY, _query_reshape_kernel->window(), query_reshape_pack);
    //const auto query_split_dimension = _query_reshape_kernel->get_split_dimension();
    /*
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif
#ifdef MEASURE_TIME
    auto   end_time  = std::chrono::high_resolution_clock::now();
    double cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt",std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "query_reshape cost: " << cost_time << std::endl;
#endif
*/

    ITensorPack query_permute_pack{ { ACL_SRC, reshaped_query.get() }, { ACL_DST, permuted_query.get() } };
    _query_permute_func->run(query_permute_pack);
    /*
#ifdef MEASURE_TIME
    start_time = std::chrono::high_resolution_clock::now();
#endif
#ifdef MEASURE_TIME
    end_time  = std::chrono::high_resolution_clock::now();
    cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    measure_out.precision(5);
    measure_out << std::scientific << "query_permute_func cost: " << cost_time << std::endl;
#endif
*/

    // Run Key multi-Head reshape
    ITensorPack key_reshape_pack{ { ACL_SRC_0, key_cpu_buffer_aux.get() }, { ACL_DST, reshaped_key.get() } };
    NEScheduler::get().schedule_op(_key_reshape_kernel.get(), Window::DimY, _key_reshape_kernel->window(), key_reshape_pack);
    //const auto key_split_dimension = _key_reshape_kernel->get_split_dimension();
    /*
#ifdef MEASURE_TIME
    start_time = std::chrono::high_resolution_clock::now();
#endif
#ifdef MEASURE_TIME
    end_time  = std::chrono::high_resolution_clock::now();
    cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    measure_out.precision(5);
    measure_out << std::scientific << "key_reshape cost: " << cost_time << std::endl;
#endif
*/

    ITensorPack key_permute_pack{ { ACL_SRC, reshaped_key.get() }, { ACL_DST, permuted_key.get() } };
    _key_permute_func->run(key_permute_pack);
    /*
#ifdef MEASURE_TIME
    start_time = std::chrono::high_resolution_clock::now();
#endif
#ifdef MEASURE_TIME
    end_time  = std::chrono::high_resolution_clock::now();
    cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    measure_out.precision(5);
    measure_out << std::scientific << "key_permute_func cost: " << cost_time << std::endl;
#endif
*/

    ITensorPack key_transpose_pack{ { ACL_SRC, permuted_key.get() }, { ACL_DST, transposed_key.get() } };
    _key_transpose_func->run(key_transpose_pack);

    // Run Value multi-Head reshape
    ITensorPack value_reshape_pack{ { ACL_SRC_0, value_cpu_buffer_aux.get() }, { ACL_DST, reshaped_value.get() } };
    NEScheduler::get().schedule_op(_value_reshape_kernel.get(), Window::DimY, _value_reshape_kernel->window(), value_reshape_pack);
    //const auto value_split_dimension = _value_reshape_kernel->get_split_dimension();
    /*
#ifdef MEASURE_TIME
    start_time = std::chrono::high_resolution_clock::now();
#endif
#ifdef MEASURE_TIME
    end_time  = std::chrono::high_resolution_clock::now();
    cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    measure_out.precision(5);
    measure_out << std::scientific << "value_reshape cost: " << cost_time << std::endl;
#endif
*/

    ITensorPack value_permute_pack{ { ACL_SRC, reshaped_value.get() }, { ACL_DST, permuted_value.get() } };
    _value_permute_func->run(value_permute_pack);
    /*
#ifdef MEASURE_TIME
    start_time = std::chrono::high_resolution_clock::now();
#endif
#ifdef MEASURE_TIME
    end_time  = std::chrono::high_resolution_clock::now();
    cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    measure_out.precision(5);
    measure_out << std::scientific << "value_permute_func cost: " << cost_time << std::endl;
#endif
*/

    // Run interleave kernel
    ITensorPack interleave_pack{ { ACL_SRC, permuted_query.get() }, { ACL_DST, interleaved_query.get() } };
    NEScheduler::get().schedule_op(_query_interleave_kernel.get(), Window::DimY, _query_interleave_kernel->window(),
                                   interleave_pack);

    // Run transpose1xw kernel
    ITensorPack transpose_pack{ { ACL_SRC, transposed_key.get() }, { ACL_DST, transposed1xw_key.get() } };
    NEScheduler::get().schedule_op(_key_transpose1xW_kernel.get(), Window::DimY,
                                   _key_transpose1xW_kernel->window(), transpose_pack);

    // Run matrix multiply compute multi-head attention between Query and Key
    ITensorPack gemm_QK_pack{ { ACL_SRC_0, interleaved_query.get() }, { ACL_SRC_1, transposed1xw_key.get() }, { ACL_DST, scaled_query_key.get() } };
    NEScheduler::get().schedule_op(_product_mm_kernel.get(), Window::DimZ, _product_mm_kernel->window(), gemm_QK_pack);
    /*
#ifdef MEASURE_TIME
    start_time = std::chrono::high_resolution_clock::now();
#endif
#ifdef MEASURE_TIME
    end_time  = std::chrono::high_resolution_clock::now();
    cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    measure_out.precision(5);
    measure_out << std::scientific << "MMUL QK cost: " << cost_time << std::endl;
#endif
*/

    if (_is_masked)
    {
        CpuAuxTensorHandler masked_scaled_kq(offset_int_vec(Mask), _masked_scaled_kq, tensors);
#ifdef MEASURE_TIME
    start_time = std::chrono::high_resolution_clock::now();
#endif
        _mask = create_mask(&_mask_info);
        ITensorPack masking_pack{{ACL_SRC_0, scaled_query_key.get()}, {ACL_SRC_1, _mask.get()}, {ACL_DST, masked_scaled_kq.get()}};
        NEScheduler::get().schedule_op(_masking_kernel.get(),Window::DimZ,_masking_kernel->window(),masking_pack);
        scaled_query_key.get()->copy_from(*masked_scaled_kq.get());
#ifdef MEASURE_TIME
    end_time  = std::chrono::high_resolution_clock::now();
    cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    measure_out.precision(5);
    measure_out << std::scientific << "CpuMasking cost: " << cost_time << std::endl;
#endif
    }

    ITensorPack softmax_pack = {{ACL_SRC, scaled_query_key.get()}, {ACL_DST, softmaxed_product.get()}};
    _softmax_func->run(softmax_pack);

    // Run interleave kernel
    ITensorPack interleave_product_pack{ { ACL_SRC, softmaxed_product.get() }, { ACL_DST, interleaved_product.get() } };
    NEScheduler::get().schedule_op(_product_interleave_kernel.get(), Window::DimY, _product_interleave_kernel->window(),
                                   interleave_product_pack);

    // Run transpose1xw kernel
    ITensorPack transpose_value_pack{ { ACL_SRC, permuted_value.get() }, { ACL_DST, transposed1xW_value.get() } };
    NEScheduler::get().schedule_op(_value_transpose1xW_kernel.get(), Window::DimY,
                                   _value_transpose1xW_kernel->window(), transpose_value_pack);

    // Run matrix multiply compute multi-head attention between Query and Key
    ITensorPack gemm_context_pack{ { ACL_SRC_0, interleaved_product.get() }, { ACL_SRC_1, transposed1xW_value.get() }, { ACL_DST, gemmed_context.get() } };
    NEScheduler::get().schedule_op(_context_mm_kernel.get(), Window::DimZ, _context_mm_kernel->window(), gemm_context_pack);
    /*
#ifdef MEASURE_TIME
    start_time = std::chrono::high_resolution_clock::now();
#endif
#ifdef MEASURE_TIME
    end_time  = std::chrono::high_resolution_clock::now();
    cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    measure_out.precision(5);
    measure_out << std::scientific << "MMUL CV cost: " << cost_time << std::endl;
#endif
*/

    // Concat all attention head together
    ITensorPack concat_permute_pack{ { ACL_SRC, gemmed_context.get() }, { ACL_DST, permuted_concat.get() } };
    _concat_permute_func->run(concat_permute_pack);
    /*
#ifdef MEASURE_TIME
    start_time = std::chrono::high_resolution_clock::now();
#endif
#ifdef MEASURE_TIME
    end_time  = std::chrono::high_resolution_clock::now();
    cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    measure_out.precision(5);
    measure_out << std::scientific << "concat_permute_func cost: " << cost_time << std::endl;
#endif
*/

    if (output->buffer() == nullptr) 
    {
        ICLTensor * output_cl = static_cast<ICLTensor *>(output);
        output_cl->map(CLScheduler::get().queue());
    }

    ITensorPack concat_reshape_pack{ { ACL_SRC_0, permuted_concat.get() }, { ACL_DST, output} };
    NEScheduler::get().schedule_op(_concat_reshape_kernel.get(), Window::DimY, _concat_reshape_kernel->window(), concat_reshape_pack);
    

    //const auto concat_split_dimension = _concat_reshape_kernel->get_split_dimension();
    /*
#ifdef MEASURE_TIME
    start_time = std::chrono::high_resolution_clock::now();
#endif
#ifdef MEASURE_TIME
    end_time  = std::chrono::high_resolution_clock::now();
    cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    measure_out.precision(5);
    measure_out << std::scientific << "concat_reshape cost: " << cost_time << std::endl;
    measure_out.close();
#endif
*/
}

experimental::MemoryRequirements CpuScaleDotProduction::workspace() const
{
    return _aux_mem;
}

} // namespace cpu
} // namespace arm_compute
