#include "arm_compute/runtime/NEON/functions/NELayerNormLayer.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/Tensor.h"

#include "src/common/utils/Log.h"
#include "src/cpu/operators/CpuLayerNorm.h"

#ifdef MEASURE_TIME
#include <chrono>
#include <fstream>
#endif

namespace arm_compute
{

struct  NELayerNormLayer::Impl
{
    const ITensor                       *src{nullptr};
    ITensor                             *dst{nullptr};
    std::unique_ptr<cpu::CpuLayerNorm>  op{nullptr};
};

NELayerNormLayer::NELayerNormLayer() : _impl(std::make_unique<Impl>())
{
}
NELayerNormLayer::~NELayerNormLayer() = default;

void NELayerNormLayer::configure(const ITensor *input,
                              ITensor *output,
                              const LayerNormLayerInfo& LayerNorm_info)
{
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_LOG_PARAMS(input, output);

    _impl->src      = input;
    _impl->dst      = output;

    _impl->op = std::make_unique<cpu::CpuLayerNorm>();
    _impl->op->configure(input->info(), output->info(), LayerNorm_info);

#ifdef MEASURE_TIME
    auto   end_time  = std::chrono::high_resolution_clock::now();
    double cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt",std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "NELayerNormLayer::configure cost: " << cost_time << std::endl;
    measure_out.close();
#endif
}

Status NELayerNormLayer::validate(const ITensor *input,
                                  ITensor *output, 
                                  const LayerNormLayerInfo& LayerNorm_info)
{
    ARM_COMPUTE_UNUSED(LayerNorm_info);
    return cpu::CpuLayerNorm::validate(input->info(), output->info(), LayerNorm_info);
}

void NELayerNormLayer::run()
{
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

    ITensorPack pack;

    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    
    _impl->op->run(pack);

#ifdef MEASURE_TIME
    auto   end_time  = std::chrono::high_resolution_clock::now();
    double cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt",std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "NELayerNormLayer::run cost: " << cost_time << std::endl;
    measure_out.close();
#endif

}

} // namespace arm_compute
