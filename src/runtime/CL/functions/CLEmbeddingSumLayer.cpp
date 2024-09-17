#include "arm_compute/runtime/CL/functions/CLEmbeddingSumLayer.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Validate.h"

#include "src/core/CL/ICLKernel.h"
#include "src/gpu/cl/operators/ClEmbedSum.h"

#ifdef MEASURE_TIME
#include <chrono>
#include <fstream>
#endif

namespace arm_compute
{

struct CLEmbeddingSumLayer::Impl
{
    const ICLTensor                    *token{ nullptr };
    const ICLTensor                    *segment{ nullptr };
    const ICLTensor                    *position{ nullptr };
    ICLTensor                          *dst{ nullptr };
    IRuntimeContext                    *ctx{ nullptr };
    std::unique_ptr<opencl::ClEmbedSum> op{ nullptr };
};

CLEmbeddingSumLayer::CLEmbeddingSumLayer()
    : _impl(std::make_unique<Impl>())
{
}

CLEmbeddingSumLayer::~CLEmbeddingSumLayer() = default;

void CLEmbeddingSumLayer::configure(ICLTensor                *token,
                                    ICLTensor                *segment,
                                    ICLTensor                *position,
                                    ICLTensor                *output,
                                    const EmbeddingLayerInfo &emb_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), token, segment, position, output, emb_info);
}

void CLEmbeddingSumLayer::configure(const CLCompileContext   &compile_context,
                                    ICLTensor                *token,
                                    ICLTensor                *segment,
                                    ICLTensor                *position,
                                    ICLTensor                *output,
                                    const EmbeddingLayerInfo &emb_info)
{
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

    _impl->token    = token;
    _impl->segment  = segment;
    _impl->position = position;
    _impl->dst      = output;

    _impl->op = std::make_unique<opencl::ClEmbedSum>();
    _impl->op->configure(compile_context,
                         token->info(),
                         segment->info(),
                         position->info(),
                         output->info(),
                         emb_info);

#ifdef MEASURE_TIME
    auto          end_time  = std::chrono::high_resolution_clock::now();
    double        cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt", std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "CLEmbeddingSumLayer::configure cost: " << cost_time << std::endl;
    measure_out.close();
#endif
}

void CLEmbeddingSumLayer::prepare()
{
}

void CLEmbeddingSumLayer::run()
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
    auto          end_time  = std::chrono::high_resolution_clock::now();
    double        cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt", std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "CLEmbeddingSumLayer::run cost: " << cost_time << std::endl;
    measure_out.close();
#endif
}

} // namespace arm_compute