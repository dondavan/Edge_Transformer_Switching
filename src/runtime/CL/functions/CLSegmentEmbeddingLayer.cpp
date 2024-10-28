#include "arm_compute/runtime/CL/functions/CLSegmentEmbeddingLayer.h"

#include "arm_compute/core/Validate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/KernelDescriptors.h"

#include "src/core/CL/ICLKernel.h"
#include "src/gpu/cl/operators/ClSegmentEmbed.h"

#ifdef MEASURE_TIME
#include <chrono>
#include <fstream>
#endif

namespace arm_compute
{

struct CLSegmentEmbeddingLayer::Impl
{
    const ITensor                        *src{ nullptr };
    const ITensor                        *segment{ nullptr };
    ITensor                              *dst{ nullptr };
    IRuntimeContext                      *ctx{ nullptr };
    std::unique_ptr<opencl::ClSegmentEmbed> op{ nullptr };
};

CLSegmentEmbeddingLayer::CLSegmentEmbeddingLayer()
    : _impl(std::make_unique<Impl>())
{
}

CLSegmentEmbeddingLayer::~CLSegmentEmbeddingLayer() = default;


void CLSegmentEmbeddingLayer::configure(ITensor *input, ITensor *segment, ITensor *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, segment, output);
}

void CLSegmentEmbeddingLayer::configure(const CLCompileContext &compile_context,ITensor *input, ITensor *segment, ITensor *output)
{
    /*
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

#ifdef MEASURE_TIME
    auto   end_time  = std::chrono::high_resolution_clock::now();
    double cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt",std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific << "CLSegmentEmbeddingLayer::configure cost: " << cost_time << std::endl;
    measure_out.close();
#endif
    */
    _impl->src     = input;
    _impl->segment = segment;
    _impl->dst     = output;
    
    _impl->op = std::make_unique<opencl::ClSegmentEmbed>();
    _impl->op->configure(compile_context, _impl->src->info(), _impl->segment->info(), _impl->dst->info());
}

void CLSegmentEmbeddingLayer::prepare()
{
}

void CLSegmentEmbeddingLayer::run()
{
    /*
#ifdef MEASURE_TIME
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

#ifdef MEASURE_TIME
    auto   end_time  = std::chrono::high_resolution_clock::now();
    double cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    std::ofstream measure_out("measure_output.txt",std::ios::app);
    measure_out.precision(5);
    measure_out << std::scientific <<  "CLSegmentEmbeddingLayer::run cost: " << cost_time << std::endl;
    measure_out.close();
#endif
    */

    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC_0, _impl->src);
    pack.add_tensor(TensorType::ACL_SRC_1, _impl->segment);
    pack.add_tensor(TensorType::ACL_DST, _impl->dst);
    _impl->op->run(pack);
}

} // namespace arm_compute