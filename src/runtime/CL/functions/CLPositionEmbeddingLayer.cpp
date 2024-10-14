#include "arm_compute/runtime/CL/functions/CLPositionEmbeddingLayer.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Validate.h"

#include "src/common/utils/Log.h"
#include "src/core/CL/ICLKernel.h"
#include "src/gpu/cl/operators/ClPositionEmbed.h"

#ifdef MEASURE_TIME
#include <chrono>
#include <fstream>
#endif

namespace arm_compute
{

struct CLPositionEmbeddingLayer::Impl
{
    const ICLTensor                         *src{ nullptr };
    const ICLTensor                         *position{ nullptr };
    ICLTensor                               *dst{ nullptr };
    IRuntimeContext                         *ctx{ nullptr };
    std::unique_ptr<opencl::ClPositionEmbed> op{ nullptr };
};

CLPositionEmbeddingLayer::CLPositionEmbeddingLayer()
    : _impl(std::make_unique<Impl>())
{
}

CLPositionEmbeddingLayer::~CLPositionEmbeddingLayer() = default;

void CLPositionEmbeddingLayer::configure(ICLTensor *input,
                                         ICLTensor *position,
                                         ICLTensor *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, position, output);
}

void CLPositionEmbeddingLayer::configure(const CLCompileContext &compile_context,
                                         ICLTensor              *input,
                                         ICLTensor              *position,
                                         ICLTensor              *output)
{
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

    _impl->src      = input;
    _impl->position = position;
    _impl->dst      = output;

    _impl->op = std::make_unique<opencl::ClPositionEmbed>();
    _impl->op->configure(compile_context, _impl->src->info(), _impl->position->info(), _impl->dst->info());

#ifdef MEASURE_TIME
    auto          end_time  = std::chrono::high_resolution_clock::now();
    double        cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt", std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "CLPositionEmbeddingLayer::configure cost: " << cost_time << std::endl;
    measure_out.close();
#endif
}

void CLPositionEmbeddingLayer::prepare()
{
}

void CLPositionEmbeddingLayer::run()
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
    auto          end_time  = std::chrono::high_resolution_clock::now();
    double        cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt", std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "CLPositionEmbeddingLayer::run cost: " << cost_time << std::endl;
    measure_out.close();
#endif
}

} // namespace arm_compute