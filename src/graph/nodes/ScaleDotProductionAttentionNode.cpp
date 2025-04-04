#include "arm_compute/graph/nodes/ScaleDotProductionAttentionNode.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

namespace arm_compute
{
namespace graph
{
ScaleDotProductionAttentionNode::ScaleDotProductionAttentionNode(ScaleDotProductionLayerInfo sdpa_info) : _sdpa_info(sdpa_info)
{
    _input_edges.resize(3, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

const ScaleDotProductionLayerInfo& ScaleDotProductionAttentionNode::sdpa_info() const
{
    return _sdpa_info;
}

bool ScaleDotProductionAttentionNode::forward_descriptors()
{
    if ((input_id(0) != NullTensorID) && (output_id(0) != NullTensorID))
    {
        Tensor *dst = output(0);
        ARM_COMPUTE_ERROR_ON(dst == nullptr);
        dst->desc() = configure_output(0);
        return true;
    }
    return false;
}

TensorDescriptor ScaleDotProductionAttentionNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    const Tensor *src = input(0);
    ARM_COMPUTE_ERROR_ON(src == nullptr);

    TensorDescriptor output_desc = src->desc();
    return src->desc();
}

NodeType ScaleDotProductionAttentionNode::type() const
{
    return NodeType::ScaleDotProductionAttentionLayer;
}

void ScaleDotProductionAttentionNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute
