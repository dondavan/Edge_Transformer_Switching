/*
 * Copyright (c) 2017-2023 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/runtime/CL/functions/CLAttentionConvolutionLayer.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/functions/CLFFTConvolutionLayer.h"

#include "src/common/utils/Log.h"
#include "src/core/CL/ICLKernel.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/gpu/cl/operators/ClConv2d.h"
#include "support/Cast.h"

namespace arm_compute
{
using namespace arm_compute::misc::shape_calculator;
using namespace arm_compute::experimental;
struct CLAttentionConvolutionLayer::Impl
{
    MemoryGroup                          memory_group{};
    std::shared_ptr<IMemoryManager>      memory_manager{};
    std::unique_ptr<opencl::IClOperator> op1{ nullptr };
    std::unique_ptr<opencl::IClOperator> op2{ nullptr };
    std::unique_ptr<opencl::IClOperator> op3{ nullptr };
    ITensorPack                          q_run_pack{};
    ITensorPack                          k_run_pack{};
    ITensorPack                          v_run_pack{};
    ITensorPack                          q_prep_pack{};
    ITensorPack                          k_prep_pack{};
    ITensorPack                          v_prep_pack{};
    WorkspaceData<CLTensor>              q_workspace{};
    WorkspaceData<CLTensor>              k_workspace{};
    WorkspaceData<CLTensor>              v_workspace{};
    experimental::MemoryRequirements     aux_mem_req{};
    std::unique_ptr<IFunction>           func1{ nullptr };
    std::unique_ptr<IFunction>           func2{ nullptr };
    std::unique_ptr<IFunction>           func3{ nullptr };
};

CLAttentionConvolutionLayer::CLAttentionConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _impl(std::make_unique<Impl>())
{
    _impl->memory_manager = std::move(memory_manager);
}

CLAttentionConvolutionLayer::~CLAttentionConvolutionLayer() = default;

void CLAttentionConvolutionLayer::configure(ITensor *query_input, ITensor *query_w, ITensor *query_b,
                                            ITensor *key_input, ITensor *key_w, ITensor *key_b,
                                            ITensor *value_input, ITensor *value_w, ITensor *value_b,
                                            ITensor *query_output, ITensor *key_output, ITensor *value_output,
                                            const PadStrideInfo       &conv_info,
                                            const WeightsInfo         &weights_info,
                                            const Size2D              &dilation,
                                            const ActivationLayerInfo &act_info,
                                            bool                       enable_fast_math,
                                            unsigned int               num_groups)
{
    configure(CLKernelLibrary::get().get_compile_context(), 
              query_input, query_w, query_b, 
              key_input, key_w, key_b,
              value_input, value_w, value_b,
              query_output, key_output, value_output, conv_info, weights_info,
              dilation, act_info, enable_fast_math, num_groups);
}

void CLAttentionConvolutionLayer::configure(const CLCompileContext &compile_context,
                                            ITensor *query_input, ITensor *query_w, ITensor *query_b,
                                            ITensor *key_input, ITensor *key_w, ITensor *key_b,
                                            ITensor *value_input, ITensor *value_w, ITensor *value_b,
                                            ITensor *query_output, ITensor *key_output, ITensor *value_output,
                                            const PadStrideInfo       &conv_info,
                                            const WeightsInfo         &weights_info,
                                            const Size2D              &dilation,
                                            const ActivationLayerInfo &act_info,
                                            bool                       enable_fast_math,
                                            unsigned int               num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(CLAttentionConvolutionLayer::validate(
        input->info(), weights->info(), ((biases != nullptr) ? biases->info() : nullptr), output->info(), conv_info,
        weights_info, dilation, act_info, enable_fast_math, num_groups));
    ARM_COMPUTE_LOG_PARAMS(input, weights, biases, output, conv_info, weights_info, dilation, act_info,
                           enable_fast_math, num_groups);

    const Conv2dInfo conv2d_info = Conv2dInfo(conv_info, dilation, act_info, enable_fast_math, num_groups);

    switch(opencl::ClConv2d::get_convolution_method(query_input->info(), query_w->info(), query_output->info(), conv2d_info,
                                                    weights_info, CLScheduler::get().target()))
    {
        case ConvolutionMethod::WINOGRAD:
        case ConvolutionMethod::DIRECT:
        case ConvolutionMethod::INDIRECT:
        case ConvolutionMethod::GEMM:
        {
            auto f1 = std::make_unique<opencl::ClConv2d>();
            auto f2 = std::make_unique<opencl::ClConv2d>();
            auto f3 = std::make_unique<opencl::ClConv2d>();
            f1->configure(compile_context, query_input->info(), query_w->info(), ((query_b != nullptr) ? query_b->info() : nullptr),
                         query_output->info(), conv2d_info, weights_info);
            f2->configure(compile_context, key_input->info(),key_w->info(), ((key_b != nullptr) ? key_b->info() : nullptr),
                         key_output->info(), conv2d_info, weights_info);
            f3->configure(compile_context, value_input->info(), value_w->info(), ((value_b != nullptr) ? value_b->info() : nullptr),
                         value_output->info(), conv2d_info, weights_info);
            _impl->op1 = std::move(f1);
            _impl->op2 = std::move(f2);
            _impl->op3 = std::move(f3);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported.");
            break;
    }

    if(_impl->op1)
    {
        _impl->memory_group = MemoryGroup(std::move(_impl->memory_manager));
        _impl->aux_mem_req  = _impl->op1->workspace();
        _impl->q_run_pack     = { { ACL_SRC_0, query_input }, { ACL_SRC_1, query_w }, { ACL_SRC_2, query_b }, { ACL_DST, query_output } };
        _impl->q_prep_pack    = { { ACL_SRC_1, query_w }, { ACL_SRC_2, query_b } };
        _impl->q_workspace =
            manage_workspace<CLTensor>(_impl->aux_mem_req, _impl->memory_group, _impl->q_run_pack, _impl->q_prep_pack);
    }
    if(_impl->op2)
    {
        _impl->memory_group = MemoryGroup(std::move(_impl->memory_manager));
        _impl->aux_mem_req  = _impl->op2->workspace();
        _impl->k_run_pack     = { { ACL_SRC_0, key_input }, { ACL_SRC_1, key_w }, { ACL_SRC_2, key_b }, { ACL_DST, key_output } };
        _impl->k_prep_pack    = { { ACL_SRC_1, key_w }, { ACL_SRC_2, key_b } };
        _impl->k_workspace =
            manage_workspace<CLTensor>(_impl->aux_mem_req, _impl->memory_group, _impl->k_run_pack, _impl->k_prep_pack);
    }
    if(_impl->op3)
    {
        _impl->memory_group = MemoryGroup(std::move(_impl->memory_manager));
        _impl->aux_mem_req  = _impl->op3->workspace();
        _impl->v_run_pack     = { { ACL_SRC_0, value_input }, { ACL_SRC_1, value_w }, { ACL_SRC_2, value_b }, { ACL_DST, value_output } };
        _impl->v_prep_pack    = { { ACL_SRC_1, value_w }, { ACL_SRC_2, value_b } };
        _impl->v_workspace =
            manage_workspace<CLTensor>(_impl->aux_mem_req, _impl->memory_group, _impl->v_run_pack, _impl->v_prep_pack);
    }
}

Status CLAttentionConvolutionLayer::validate(const ITensorInfo         *input,
                                             const ITensorInfo         *weights,
                                             const ITensorInfo         *biases,
                                             const ITensorInfo         *output,
                                             const PadStrideInfo       &conv_info,
                                             const WeightsInfo         &weights_info,
                                             const Size2D              &dilation,
                                             const ActivationLayerInfo &act_info,
                                             bool                       enable_fast_math,
                                             unsigned int               num_groups)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!weights->are_values_constant(), "Dynamic weights are not supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((num_groups != 1) && (input->data_layout() != DataLayout::NCHW),
                                    "Grouping (num_groups != 1) with NHWC data layout is not supported");

    const GPUTarget  gpu_target  = CLScheduler::get().target();
    const Conv2dInfo conv2d_info = Conv2dInfo(conv_info, dilation, act_info, enable_fast_math, num_groups);

    switch(opencl::ClConv2d::get_convolution_method(input, weights, output, conv2d_info, weights_info, gpu_target))
    {
        case ConvolutionMethod::WINOGRAD:
        case ConvolutionMethod::DIRECT:
        case ConvolutionMethod::INDIRECT:
        case ConvolutionMethod::GEMM:
        {
            ARM_COMPUTE_RETURN_ON_ERROR(
                opencl::ClConv2d::validate(input, weights, biases, output, conv2d_info, weights_info));
            break;
        }
        case ConvolutionMethod::FFT:
        {
            // Validate FFT-based convolution layer
            ARM_COMPUTE_RETURN_ON_ERROR(CLFFTConvolutionLayer::validate(input, weights, nullptr, output, conv_info,
                                                                        act_info, enable_fast_math));
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported.");
            break;
    }

    return Status{};
}

ConvolutionMethod CLAttentionConvolutionLayer::get_convolution_method(const ITensorInfo         *input,
                                                                      const ITensorInfo         *weights,
                                                                      const ITensorInfo         *output,
                                                                      const PadStrideInfo       &conv_info,
                                                                      const WeightsInfo         &weights_info,
                                                                      const ActivationLayerInfo &act_info,
                                                                      const GPUTarget            gpu_target,
                                                                      const Size2D              &dilation,
                                                                      bool                       enable_fast_math)
{
    const Conv2dInfo conv2d_info = Conv2dInfo(conv_info, dilation, act_info, enable_fast_math, 1);
    return opencl::ClConv2d::get_convolution_method(input, weights, output, conv2d_info, weights_info, gpu_target);
}

void CLAttentionConvolutionLayer::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_impl->memory_group);

    if(_impl->func1)
    {
        _impl->func1->run();
    }
    else
    {
        _impl->op1->run(_impl->q_run_pack);
    }

    if(_impl->func2)
    {
        _impl->func2->run();
    }
    else
    {
        _impl->op2->run(_impl->k_run_pack);
    }

    if(_impl->func3)
    {
        _impl->func3->run();
    }
    else
    {
        _impl->op3->run(_impl->v_run_pack);
    }
}

void CLAttentionConvolutionLayer::prepare()
{
    if(_impl->func1)
    {
        _impl->func1->prepare();
    }
    else
    {
        _impl->op1->prepare(_impl->q_prep_pack);

        // Release temporary tensors that are only used in prepare stage
        release_temporaries(_impl->aux_mem_req, _impl->q_workspace);
    }

    if(_impl->func2)
    {
        _impl->func2->prepare();
    }
    else
    {
        _impl->op2->prepare(_impl->k_prep_pack);

        // Release temporary tensors that are only used in prepare stage
        release_temporaries(_impl->aux_mem_req, _impl->k_workspace);
    }

    if(_impl->func3)
    {
        _impl->func3->prepare();
    }
    else
    {
        _impl->op3->prepare(_impl->v_prep_pack);

        // Release temporary tensors that are only used in prepare stage
        release_temporaries(_impl->aux_mem_req, _impl->v_workspace);
    }
}
} // namespace arm_compute
