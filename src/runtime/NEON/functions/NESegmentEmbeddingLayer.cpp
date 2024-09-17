#include "arm_compute/runtime/NEON/functions/NESegmentEmbeddingLayer.h"

#include "arm_compute/core/Validate.h"

#include "src/cpu/operators/CpuSegmentEmbed.h"

#ifdef MEASURE_TIME
#include <chrono>
#include <fstream>
#endif

namespace arm_compute
{

struct NESegmentEmbeddingLayer::Impl
{
    const ITensor                        *src{ nullptr };
    const ITensor                        *segment{ nullptr };
    ITensor                              *dst{ nullptr };
    IRuntimeContext                      *ctx{ nullptr };
    std::unique_ptr<cpu::CpuSegmentEmbed> op{ nullptr };
};

NESegmentEmbeddingLayer::NESegmentEmbeddingLayer()
    : _impl(std::make_unique<Impl>())
{
}

NESegmentEmbeddingLayer::~NESegmentEmbeddingLayer() = default;

void NESegmentEmbeddingLayer::configure(ITensor *input, ITensor *segment, ITensor *output)
{
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

    _impl->src     = input;
    _impl->segment = segment;
    _impl->dst     = output;

    _impl->op = std::make_unique<cpu::CpuSegmentEmbed>();
    _impl->op->configure(_impl->src->info(), _impl->segment->info(), _impl->dst->info());

#ifdef MEASURE_TIME
    auto   end_time  = std::chrono::high_resolution_clock::now();
    double cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt",std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "NESegmentEmbeddingLayer::configure cost: " << cost_time << std::endl;
    measure_out.close();
#endif
}

void NESegmentEmbeddingLayer::prepare()
{
}

void NESegmentEmbeddingLayer::run()
{
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif


    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->segment);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);

#ifdef MEASURE_TIME
    auto   end_time  = std::chrono::high_resolution_clock::now();
    double cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt",std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific <<  "NESegmentEmbeddingLayer::run cost: " << cost_time << std::endl;
    measure_out.close();
#endif
}

} // namespace arm_compute