#include "arm_compute/runtime/NEON/functions/NEPositionEmbeddingLayer.h"

#include "arm_compute/core/Validate.h"

#include "src/cpu/operators/CpuPositionEmbed.h"

#ifdef MEASURE_TIME
#include <chrono>
#include <fstream>
#endif

namespace arm_compute
{

struct NEPositionEmbeddingLayer::Impl
{
    const ITensor                         *src{ nullptr };
    const ITensor                         *position{ nullptr };
    ITensor                               *dst{ nullptr };
    IRuntimeContext                       *ctx{ nullptr };
    std::unique_ptr<cpu::CpuPositionEmbed> op{ nullptr };
};

NEPositionEmbeddingLayer::NEPositionEmbeddingLayer()
    : _impl(std::make_unique<Impl>())
{
}

NEPositionEmbeddingLayer::~NEPositionEmbeddingLayer() = default;

void NEPositionEmbeddingLayer::configure(ITensor *input, ITensor *position, ITensor *output)
{
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

    _impl->src      = input;
    _impl->position = position;
    _impl->dst      = output;

    _impl->op = std::make_unique<cpu::CpuPositionEmbed>();
    _impl->op->configure(_impl->src->info(), _impl->position->info(), _impl->dst->info());

#ifdef MEASURE_TIME
    auto   end_time  = std::chrono::high_resolution_clock::now();
    double cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt",std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "NEPositionEmbeddingLayer::configure cost: " << cost_time << std::endl;
    measure_out.close();
#endif
}

void NEPositionEmbeddingLayer::prepare()
{
}

void NEPositionEmbeddingLayer::run()
{
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->position);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);


#ifdef MEASURE_TIME
    auto   end_time  = std::chrono::high_resolution_clock::now();
    double cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt",std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "NEPositionEmbeddingLayer::run cost: " << cost_time << std::endl;
    measure_out.close();
#endif
}

} // namespace arm_compute