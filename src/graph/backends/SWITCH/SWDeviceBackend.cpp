/*
 * Copyright (c) 2018-2021,2023 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/graph/backends/SWITCH/SWDeviceBackend.h"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/graph/backends/BackendRegistrar.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/INode.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/runtime/Allocator.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/IWeightsManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"
#include "arm_compute/runtime/OffsetLifetimeManager.h"
#include "arm_compute/runtime/PoolManager.h"
#include "arm_compute/runtime/Scheduler.h"

namespace arm_compute
{
namespace graph
{
namespace backends
{
/** Register SWITCHING backend */
static detail::BackendRegistrar<SWDeviceBackend> SWDeviceBackend_registrar(Target::SWITCH);

SWDeviceBackend::SWDeviceBackend() : _allocator()
{
}

void SWDeviceBackend::initialize_backend()
{
    //Nothing to do
}

void SWDeviceBackend::release_backend_context(GraphContext &ctx)
{
    //Nothing to do
    ARM_COMPUTE_UNUSED(ctx);
}

void SWDeviceBackend::setup_backend_context(GraphContext &ctx)
{
    //Nothing to do
    ARM_COMPUTE_UNUSED(ctx);
}

bool SWDeviceBackend::is_backend_supported()
{
    return true;
}

IAllocator *SWDeviceBackend::backend_allocator()
{
    return &_allocator;
}

std::unique_ptr<ITensorHandle> SWDeviceBackend::create_tensor(const Tensor &tensor)
{
    //Nothing to do
    ARM_COMPUTE_UNUSED(tensor);

    return nullptr;
}

std::unique_ptr<ITensorHandle>
SWDeviceBackend::create_subtensor(ITensorHandle *parent, TensorShape shape, Coordinates coords, bool extend_parent)
{
    //Nothing to do
    ARM_COMPUTE_UNUSED(parent,shape,coords,extend_parent);

    return nullptr;
}

std::unique_ptr<arm_compute::IFunction> SWDeviceBackend::configure_node(INode &node, GraphContext &ctx)
{
    //Nothing to do
    ARM_COMPUTE_UNUSED(node,ctx);

    return nullptr;
}

arm_compute::Status SWDeviceBackend::validate_node(INode &node)
{
    ARM_COMPUTE_UNUSED(node);

    return Status{};
}

std::shared_ptr<arm_compute::IMemoryManager> SWDeviceBackend::create_memory_manager(MemoryManagerAffinity affinity)
{
    ARM_COMPUTE_UNUSED(affinity);

    return nullptr;
}

std::shared_ptr<arm_compute::IWeightsManager> SWDeviceBackend::create_weights_manager()
{
    // nop

    return nullptr;
}

void SWDeviceBackend::sync()
{
    // nop
}
} // namespace backends
} // namespace graph
} // namespace arm_compute
