#include "arm_compute/runtime/CL/functions/CLScaleDotProductionAttentionLayer.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

#include "src/core/CL/ICLKernel.h"
#include "src/gpu/cl/operators/ClScaleDotProduction.h"

#ifdef MEASURE_TIME
#include <chrono>
#include <fstream>
#endif

namespace arm_compute
{

struct CLScaleDotProductionAttentionLayer::Impl
{
    ITensorPack scale_dot_pack{};

    IRuntimeContext *ctx{ nullptr };

    std::unique_ptr<opencl::ClScaleDotProduction> scale_dot_production_op{ nullptr };

    bool is_prepared{ false };
};

CLScaleDotProductionAttentionLayer::CLScaleDotProductionAttentionLayer()
    : _impl(std::make_unique<Impl>())
{
}

CLScaleDotProductionAttentionLayer::~CLScaleDotProductionAttentionLayer() = default;

void CLScaleDotProductionAttentionLayer::configure(const ICLTensor                            *query,
                                                   const ICLTensor                            *key,
                                                   const ICLTensor                            *value,
                                                   ICLTensor                                  *output,
                                                   const ScaleDotProductionAttentionLayerInfo &info)
{
    configure(CLKernelLibrary::get().get_compile_context(), query, key, value, output, info);
}
void CLScaleDotProductionAttentionLayer::configure(const CLCompileContext                     &compile_context,
                                                   const ICLTensor                            *query,
                                                   const ICLTensor                            *key,
                                                   const ICLTensor                            *value,
                                                   ICLTensor                                  *output,
                                                   const ScaleDotProductionAttentionLayerInfo &info)
{
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif
    
    std::cout << "CLScaleDotProductionAttentionLayer::configure start" << std::endl;

    /* Scale dot production of key and query */
    _impl->scale_dot_production_op = std::make_unique<opencl::ClScaleDotProduction>();
    _impl->scale_dot_production_op->configure(compile_context, query->info(), key->info(), value->info(), output->info(), info);
    _impl->scale_dot_pack = { { ACL_SRC_0, query }, { ACL_SRC_1, key }, { ACL_SRC_2, value }, { ACL_DST, output } };

    std::cout << "CLScaleDotProductionAttentionLayer::configure end" << std::endl;

#ifdef MEASURE_TIME
    auto          end_time  = std::chrono::high_resolution_clock::now();
    double        cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt", std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "CLScaleDotProductionAttentionLayer::configure cost: " << cost_time << std::endl;
    measure_out.close();
#endif
}

void CLScaleDotProductionAttentionLayer::run()
{
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

    ITensorPack pack;

    _impl->scale_dot_production_op->run(_impl->scale_dot_pack);

#ifdef MEASURE_TIME
    auto          end_time  = std::chrono::high_resolution_clock::now();
    double        cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt", std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "CLScaleDotProductionAttentionLayer::run cost: " << cost_time << std::endl;
    measure_out.close();
#endif
}

} // namespace arm_compute