#include "arm_compute/graph/nodes/AttentionLinearNode.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

namespace arm_compute
{
namespace graph
{
AttentionLinearNode::AttentionLinearNode(LinearLayerInfo info): _linear_info(std::move(info))
{
    _input_edges.resize(9, EmptyEdgeID); // Input, weight, bias * QKV
    _outputs.resize(3, NullTensorID);    // QKV
}

const LinearLayerInfo& AttentionLinearNode::linear_info() const
{
    return _linear_info;
}

bool AttentionLinearNode::forward_descriptors()
{
    
    if ((input_id(0) != NullTensorID) && (output_id(0) != NullTensorID) 
        && (input_id(3) != NullTensorID) && (output_id(1) != NullTensorID) 
        && (input_id(6) != NullTensorID) && (output_id(2) != NullTensorID))
    {
        // Query
        Tensor *dst0 = output(0);
        ARM_COMPUTE_ERROR_ON(dst0 == nullptr);
        dst0->desc() = configure_output(0);

        // Key
        Tensor *dst1 = output(1);
        ARM_COMPUTE_ERROR_ON(dst1 == nullptr);
        dst1->desc() = configure_output(1);

        // Value
        Tensor *dst2 = output(2);
        ARM_COMPUTE_ERROR_ON(dst2 == nullptr);
        dst2->desc() = configure_output(2);
        return true;
    }
    return false;
}


TensorDescriptor AttentionLinearNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    const Tensor *src = input(0);
    ARM_COMPUTE_ERROR_ON(src == nullptr);

    TensorDescriptor output_desc = src->desc();
    return src->desc();
}


NodeType AttentionLinearNode::type() const
{
    return NodeType::AttentionLinearLayer;
}

void AttentionLinearNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute
