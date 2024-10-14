#include "arm_compute/runtime/NEON/functions/NELinearLayer.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/Tensor.h"

#include "src/common/utils/Log.h"
#include "src/cpu/operators/CpuLinear.h"

#ifdef MEASURE_TIME
#include <chrono>
#include <fstream>
#endif

namespace arm_compute
{

struct  NELinearLayer::Impl
{
    const ITensor                      *src{nullptr};
    const ITensor                      *weight{nullptr};
    const ITensor                      *bias{nullptr};
    ITensor                            *dst{nullptr};
    std::unique_ptr<cpu::CpuLinear>    kernel{nullptr};
};

NELinearLayer::NELinearLayer() : _impl(std::make_unique<Impl>())
{
}
NELinearLayer::~NELinearLayer() = default;

void NELinearLayer::configure(const ITensor *input, 
                              const ITensor *weight, 
                              const ITensor *bias, ITensor *output, const LinearLayerInfo& linear_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_LOG_PARAMS(input, output);
    ARM_COMPUTE_UNUSED(linear_info);

#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

    _impl->src      = input;
    _impl->weight   = weight;
    _impl->bias     = bias;
    _impl->dst      = output;

    _impl->kernel = std::make_unique<cpu::CpuLinear>();
    _impl->kernel->configure(input->info(), weight->info(), bias->info(), output->info(), 1.0f, 1.0f);

#ifdef MEASURE_TIME
    auto   end_time  = std::chrono::high_resolution_clock::now();
    double cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt",std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "NELinearLayer::configure cost: " << cost_time << std::endl;
    measure_out.close();
#endif
}

Status NELinearLayer::validate(const ITensor *input, 
                              const ITensor *weight, 
                              const ITensor *bias, ITensor *output, const LinearLayerInfo& linear_info)
{
    ARM_COMPUTE_UNUSED(linear_info);
    return cpu::CpuLinear::validate(input->info(), weight->info(), bias->info(), output->info(), 1.0f, 1.0f);
}

void NELinearLayer::run()
{
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

    ITensorPack pack;

    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->weight);
    pack.add_tensor(TensorType::ACL_SRC_2, _impl->bias);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    
    _impl->kernel->run(pack);

#ifdef MEASURE_TIME
    auto   end_time  = std::chrono::high_resolution_clock::now();
    double cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt",std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "NELinearLayer::run cost: " << cost_time << std::endl;
    measure_out.close();
#endif

}

} // namespace arm_compute
