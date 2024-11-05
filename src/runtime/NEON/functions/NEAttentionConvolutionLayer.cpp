/*
 * Copyright (c) 2017-2021, 2023 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEAttentionConvolutionLayer.h"

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/DataTypeUtils.h"
#include "arm_compute/runtime/NEON/functions/NEFFTConvolutionLayer.h"

#include "src/common/utils/Log.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/operators/CpuConv2d.h"
#include "src/cpu/operators/CpuDirectConv2d.h"
#include "src/cpu/operators/CpuGemmConv2d.h"
#include "src/cpu/operators/CpuGemmDirectConv2d.h"
#include "src/cpu/operators/CpuWinogradConv2d.h"

namespace arm_compute
{
using namespace arm_compute::experimental;

struct NEAttentionConvolutionLayer::Impl
{
    MemoryGroup                        memory_group{};
    std::shared_ptr<IMemoryManager>    memory_manager{};
    std::unique_ptr<cpu::ICpuOperator> op1{ nullptr };
    std::unique_ptr<cpu::ICpuOperator> op2{ nullptr };
    std::unique_ptr<cpu::ICpuOperator> op3{ nullptr };
    ITensorPack                        run_pack{};
    ITensorPack                        prep_pack{};
    WorkspaceData<Tensor>              workspace{};
    experimental::MemoryRequirements   aux_mem_req{};
    std::unique_ptr<IFunction>         func1{ nullptr };
    std::unique_ptr<IFunction>         func2{ nullptr };
    std::unique_ptr<IFunction>         func3{ nullptr };
};

NEAttentionConvolutionLayer::NEAttentionConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _impl(std::make_unique<Impl>())
{
    _impl->memory_manager = std::move(memory_manager);
}

NEAttentionConvolutionLayer::~NEAttentionConvolutionLayer() = default;

void NEAttentionConvolutionLayer::configure(ITensor *query_input, const ITensor *query_w, const ITensor *query_b,
                                            ITensor *key_input, const ITensor *key_w, const ITensor *key_b,
                                            ITensor *value_input, const ITensor *value_w, const ITensor *value_b,
                                            ITensor *query_output, ITensor *key_output, ITensor *value_output,
                                            const PadStrideInfo       &conv_info,
                                            const WeightsInfo         &weights_info,
                                            const Size2D              &dilation,
                                            const ActivationLayerInfo &act_info,
                                            bool                       enable_fast_math,
                                            unsigned int               num_groups)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(query_input, query_w, query_output);
    ARM_COMPUTE_UNUSED(num_groups);
    ARM_COMPUTE_ERROR_THROW_ON(NEAttentionConvolutionLayer::validate(
        query_input->info(), query_w->info(), ((biases != nullptr) ? query_b->info() : nullptr), query_output->info(), conv_info,
        weights_info, dilation, act_info, enable_fast_math, num_groups));
    ARM_COMPUTE_LOG_PARAMS(query_input, query_weights, query_b, query_output, conv_info, weights_info, dilation, act_info,
                           enable_fast_math, num_groups);

    const Conv2dInfo info(conv_info, dilation, act_info, enable_fast_math, num_groups);
    switch(cpu::CpuConv2d::get_convolution_method(query_input->info(), query_w->info(), query_output->info(), conv_info,
                                                  weights_info, dilation, act_info, enable_fast_math))
    {
        case ConvolutionMethod::WINOGRAD:
        case ConvolutionMethod::GEMM:
        case ConvolutionMethod::GEMM_CONV2D:
        case ConvolutionMethod::DIRECT:
        {
            auto f1 = std::make_unique<cpu::CpuConv2d>();
            auto f2 = std::make_unique<cpu::CpuConv2d>();
            auto f3 = std::make_unique<cpu::CpuConv2d>();
            f1->configure(query_input->info(), query_w->info(), ((query_b != nullptr) ? query_b->info() : nullptr),
                         query_output->info(), 
                         conv_info, weights_info, dilation, act_info, enable_fast_math, num_groups);
            f2->configure(key_input->info(),key_w->info(), ((key_b != nullptr) ? key_b->info() : nullptr),
                         key_output->info(), 
                         conv_info, weights_info, dilation, act_info, enable_fast_math, num_groups);
            f3->configure(value_input->info(), value_w->info(), ((value_b != nullptr) ? value_b->info() : nullptr),
                         value_output->info(), 
                         conv_info, weights_info, dilation, act_info, enable_fast_math, num_groups);
            _impl->op1 = std::move(f1);
            _impl->op2 = std::move(f2);
            _impl->op3 = std::move(f3);
            break;
        }
        case ConvolutionMethod::FFT:
        {
            auto f1 = std::make_unique<NEFFTConvolutionLayer>(_impl->memory_manager);
            auto f2 = std::make_unique<NEFFTConvolutionLayer>(_impl->memory_manager);
            auto f3 = std::make_unique<NEFFTConvolutionLayer>(_impl->memory_manager);
            f1->configure(query_input, query_w, query_b, query_output, conv_info, act_info);
            f2->configure(key_input, key_w, key_b, key_output, conv_info, act_info);
            f3->configure(value_input, value_w, value_b, value_output, conv_info, act_info);
            _impl->func1 = std::move(f1);
            _impl->func2 = std::move(f2);
            _impl->func3 = std::move(f3);
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
        _impl->run_pack     = { { ACL_SRC_0, query_input }, { ACL_SRC_1, query_w }, { ACL_SRC_2, query_b }, { ACL_DST, query_output } };
        _impl->prep_pack    = { { ACL_SRC_1, query_w }, { ACL_SRC_2, query_b } };
        _impl->workspace =
            manage_workspace<Tensor>(_impl->aux_mem_req, _impl->memory_group, _impl->run_pack, _impl->prep_pack);
    }
    if(_impl->op2)
    {
        _impl->memory_group = MemoryGroup(std::move(_impl->memory_manager));
        _impl->aux_mem_req  = _impl->op2->workspace();
        _impl->run_pack     = { { ACL_SRC_0, key_input }, { ACL_SRC_1, key_w }, { ACL_SRC_2, key_b }, { ACL_DST, key_output } };
        _impl->prep_pack    = { { ACL_SRC_1, key_w }, { ACL_SRC_2, key_b } };
        _impl->workspace =
            manage_workspace<Tensor>(_impl->aux_mem_req, _impl->memory_group, _impl->run_pack, _impl->prep_pack);
    }
    if(_impl->op3)
    {
        _impl->memory_group = MemoryGroup(std::move(_impl->memory_manager));
        _impl->aux_mem_req  = _impl->op3->workspace();
        _impl->run_pack     = { { ACL_SRC_0, value_input }, { ACL_SRC_1, value_w }, { ACL_SRC_2, value_b }, { ACL_DST, value_output } };
        _impl->prep_pack    = { { ACL_SRC_1, value_w }, { ACL_SRC_2, value_b } };
        _impl->workspace =
            manage_workspace<Tensor>(_impl->aux_mem_req, _impl->memory_group, _impl->run_pack, _impl->prep_pack);
    }
}

Status NEAttentionConvolutionLayer::validate(const ITensorInfo         *input,
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
    const Conv2dInfo info(conv_info, dilation, act_info, enable_fast_math, num_groups);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!weights->are_values_constant(), "Dynamic weights are not supported");

    // Biases with dynamic values are not supported with quantized inputs.
    if(biases)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG((!biases->are_values_constant() && is_data_type_quantized(input->data_type())),
                                        "Dynamic Biases are not supported with quantized input data.");
    }

    switch(cpu::CpuConv2d::get_convolution_method(input, weights, output, conv_info, weights_info, dilation, act_info,
                                                  enable_fast_math))
    {
        case ConvolutionMethod::WINOGRAD:
        case ConvolutionMethod::GEMM:
        case ConvolutionMethod::GEMM_CONV2D:
        case ConvolutionMethod::DIRECT:
            ARM_COMPUTE_RETURN_ON_ERROR(cpu::CpuConv2d::validate(input, weights, biases, output, conv_info,
                                                                 weights_info, dilation, act_info, enable_fast_math,
                                                                 num_groups));
            break;
        case ConvolutionMethod::FFT:
            ARM_COMPUTE_RETURN_ON_ERROR(
                NEFFTConvolutionLayer::validate(input, weights, biases, output, conv_info, act_info));
            break;
        default:
            ARM_COMPUTE_ERROR("Not supported.");
            break;
    }
    return Status{};
}

ConvolutionMethod NEAttentionConvolutionLayer::get_convolution_method(const ITensorInfo         *input,
                                                                      const ITensorInfo         *weights,
                                                                      const ITensorInfo         *output,
                                                                      const PadStrideInfo       &conv_info,
                                                                      const WeightsInfo         &weights_info,
                                                                      const Size2D              &dilation,
                                                                      const ActivationLayerInfo &act_info,
                                                                      bool                       enable_fast_math)
{
    return cpu::CpuConv2d::get_convolution_method(input, weights, output, conv_info, weights_info, dilation, act_info,
                                                  enable_fast_math);
}

void NEAttentionConvolutionLayer::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_impl->memory_group);

    if(_impl->func1)
    {
        _impl->func1->run();
    }
    else
    {
        _impl->op1->run(_impl->run_pack);
    }

    if(_impl->func2)
    {
        _impl->func2->run();
    }
    else
    {
        _impl->op2->run(_impl->run_pack);
    }

    if(_impl->func3)
    {
        _impl->func3->run();
    }
    else
    {
        _impl->op3->run(_impl->run_pack);
    }
}

void NEAttentionConvolutionLayer::prepare()
{
    if(_impl->func1)
    {
        _impl->func1->prepare();
    }
    else
    {
        _impl->op1->prepare(_impl->prep_pack);

        // Release temporary tensors that are only used in prepare stage
        release_temporaries<Tensor>(_impl->aux_mem_req, _impl->workspace);
    }

    if(_impl->func2)
    {
        _impl->func2->prepare();
    }
    else
    {
        _impl->op2->prepare(_impl->prep_pack);

        // Release temporary tensors that are only used in prepare stage
        release_temporaries<Tensor>(_impl->aux_mem_req, _impl->workspace);
    }

    if(_impl->func3)
    {
        _impl->func3->prepare();
    }
    else
    {
        _impl->op3->prepare(_impl->prep_pack);

        // Release temporary tensors that are only used in prepare stage
        release_temporaries<Tensor>(_impl->aux_mem_req, _impl->workspace);
    }
}
} // namespace arm_compute
