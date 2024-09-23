#include "arm_compute/runtime/NEON/functions/NEEmbeddingSumLayer.h"

#include "arm_compute/core/Validate.h"

#include "src/cpu/operators/CpuEmbedSum.h"

#ifdef MEASURE_TIME
#include <chrono>
#include <fstream>
#endif

namespace arm_compute
{

struct NEEmbeddingSumLayer::Impl
{
    const ITensor                    *token{ nullptr };
    const ITensor                    *segment{ nullptr };
    const ITensor                    *position{ nullptr };
    ITensor                          *dst{ nullptr };
    IRuntimeContext                  *ctx{ nullptr };
    std::unique_ptr<cpu::CpuEmbedSum> op{ nullptr };
};

NEEmbeddingSumLayer::NEEmbeddingSumLayer()
    : _impl(std::make_unique<Impl>())
{
}

NEEmbeddingSumLayer::~NEEmbeddingSumLayer() = default;

void NEEmbeddingSumLayer::configure(ITensor *token, ITensor *segment, ITensor *position, ITensor *output, const EmbeddingLayerInfo &emb_info)
{
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

    _impl->token    = token;
    _impl->segment  = segment;
    _impl->position = position;
    _impl->dst      = output;

    _impl->op = std::make_unique<cpu::CpuEmbedSum>();
    _impl->op->configure(_impl->token->info(),
                         _impl->segment->info(),
                         _impl->position->info(),
                         _impl->dst->info(),
                         emb_info);

#ifdef MEASURE_TIME
    auto   end_time  = std::chrono::high_resolution_clock::now();
    double cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt",std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "NEEmbeddingSumLayer::configure cost: " << cost_time << std::endl;
    measure_out.close();
#endif
}

void NEEmbeddingSumLayer::prepare()
{
}

void NEEmbeddingSumLayer::run()
{
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->token);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->segment);
    pack.add_tensor(TensorType::ACL_SRC_2, _impl->position);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);

    _impl->op->run(pack);

#ifdef MEASURE_TIME
    auto   end_time  = std::chrono::high_resolution_clock::now();
    double cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt",std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "NEEmbeddingSumLayer::run cost: " << cost_time << std::endl;
    measure_out.close();
#endif
}

} // namespace arm_compute