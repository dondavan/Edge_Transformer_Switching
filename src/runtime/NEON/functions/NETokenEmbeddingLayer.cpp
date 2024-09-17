#include "arm_compute/runtime/NEON/functions/NETokenEmbeddingLayer.h"

#include "arm_compute/core/Validate.h"

#include "src/cpu/operators/CpuTokenEmbed.h"

#ifdef MEASURE_TIME
#include <chrono>
#include <fstream>
#endif


namespace arm_compute
{

struct NETokenEmbeddingLayer::Impl
{
    const ITensor                      *src{ nullptr };
    const ITensor                      *vocab{ nullptr };
    ITensor                            *dst{ nullptr };
    IRuntimeContext                    *ctx{ nullptr };
    std::unique_ptr<cpu::CpuTokenEmbed> op{ nullptr };
};

NETokenEmbeddingLayer::NETokenEmbeddingLayer()
    : _impl(std::make_unique<Impl>())
{
}

NETokenEmbeddingLayer::~NETokenEmbeddingLayer() = default;

void NETokenEmbeddingLayer::configure(ITensor *input, ITensor *vocab, ITensor *output, const EmbeddingLayerInfo &emb_info)
{
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif
    _impl->src   = input;
    _impl->vocab = vocab;
    _impl->dst   = output;

    _impl->op = std::make_unique<cpu::CpuTokenEmbed>();
    _impl->op->configure(_impl->src->info(), _impl->vocab->info(), _impl->dst->info(), emb_info);

#ifdef MEASURE_TIME
    auto   end_time  = std::chrono::high_resolution_clock::now();
    double cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt",std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "NETokenEmbeddingLayer::configure cost: " << cost_time << std::endl;
    measure_out.close();
#endif
}

void NETokenEmbeddingLayer::prepare()
{
}

void NETokenEmbeddingLayer::run()
{
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->vocab);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);

#ifdef MEASURE_TIME
    auto   end_time  = std::chrono::high_resolution_clock::now();
    double cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt",std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "NETokenEmbeddingLayer::run cost: " << cost_time << std::endl;
    measure_out.close();
#endif
}

} // namespace arm_compute