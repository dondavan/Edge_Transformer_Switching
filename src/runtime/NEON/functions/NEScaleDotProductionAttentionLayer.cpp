#include "arm_compute/runtime/NEON/functions/NEScaleDotProductionAttentionLayer.h"

#include "arm_compute/core/Validate.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/operators/CpuScaleDotProduction.h"
#include "src/cpu/operators/CpuGemm.h"

#ifdef MEASURE_TIME
#include <chrono>
#include <fstream>
#endif

namespace arm_compute
{

struct NEScaleDotProductionAttentionLayer::Impl
{

    MemoryGroup                         memory_group{};

    ITensorPack                         scale_dot_pack{};

    IRuntimeContext                    *ctx{nullptr};

    std::unique_ptr<cpu::CpuScaleDotProduction> scale_dot_production_op{nullptr};

    bool is_prepared{false};
};

struct NEScaleDotProductionAttentionLayer::Recurrence
{
    unsigned int recurrence_count{0};
    ITensor     *query{nullptr};
    ITensor     *key{nullptr};
    ITensor     *value{nullptr};
    ITensor     *output{nullptr};
};

NEScaleDotProductionAttentionLayer::NEScaleDotProductionAttentionLayer()
    : _impl(std::make_unique<Impl>()), _recurrence(std::make_unique<Recurrence>())
{
}

NEScaleDotProductionAttentionLayer::~NEScaleDotProductionAttentionLayer() = default;

void NEScaleDotProductionAttentionLayer::configure(ITensor *query,
                                                   ITensor *key,
                                                   ITensor *value,
                                                   ITensor *output,
                                                   const ScaleDotProductionLayerInfo& info,
                                                   int recurrence_count)
{
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif
    std::cout << "NEScaleDotProductionAttentionLayer::configure recurrence count: " << recurrence_count << std::endl;
    /* Scale dot production of key and query */
    _impl->scale_dot_production_op  = std::make_unique<cpu::CpuScaleDotProduction>();
    _impl->scale_dot_production_op->configure(query->info(),key->info(),value->info(),output->info(),info,recurrence_count);
    _impl->scale_dot_pack = {{ACL_SRC_0, query}, {ACL_SRC_1, key}, {ACL_SRC_2, value}, {ACL_DST, output}};

#ifdef MEASURE_TIME
    auto   end_time  = std::chrono::high_resolution_clock::now();
    double cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt",std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "NEScaleDotProductionAttentionLayer::configure cost: " << cost_time << std::endl;
    measure_out.close();
#endif

}

void NEScaleDotProductionAttentionLayer::run()
{
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

    ITensorPack pack;

    _impl->scale_dot_production_op->run(_impl->scale_dot_pack);
    
#ifdef MEASURE_TIME
    auto   end_time  = std::chrono::high_resolution_clock::now();
    double cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt",std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "NEScaleDotProductionAttentionLayer::run cost: " << cost_time << std::endl;
    measure_out.close();
#endif

}

} // namespace arm_compute