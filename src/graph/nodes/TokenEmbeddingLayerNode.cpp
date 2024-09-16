
#include "arm_compute/graph/nodes/TokenEmbeddingLayerNode.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INodeVisitor.h"

namespace arm_compute
{
namespace graph
{
TokenEmbeddingLayerNode::TokenEmbeddingLayerNode(EmbeddingLayerInfo info): _info(std::move(info))
{
    _input_edges.resize(2 /*0 token id input*/
                          /*1 vector const input*/, EmptyEdgeID);
    _outputs.resize(1, NullTensorID);
}

EmbeddingLayerInfo TokenEmbeddingLayerNode::token_embedding_info() const
{
    return _info;
}

TensorDescriptor TokenEmbeddingLayerNode::compute_output_descriptor(const TensorDescriptor &input_descriptor,
                                                                    const TensorDescriptor &vector_descriptor,
                                                                   EmbeddingLayerInfo emb_info)
{
    ARM_COMPUTE_UNUSED(emb_info);
    TensorDescriptor output_descriptor = vector_descriptor;
    output_descriptor.shape.set(1, input_descriptor.shape.x());

    return output_descriptor;
}

bool TokenEmbeddingLayerNode::forward_descriptors()
{
    if ((input_id(0) != NullTensorID) && input_id(1) != NullTensorID && (output_id(0) != NullTensorID))
    {
        Tensor *dst = output(0);
        ARM_COMPUTE_ERROR_ON(dst == nullptr);
        dst->desc() = configure_output(0);
        return true;
    }
    return false;
}

TensorDescriptor TokenEmbeddingLayerNode::configure_output(size_t idx) const
{
    ARM_COMPUTE_UNUSED(idx);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());

    const Tensor *src = input(0/*token id input*/);
    const Tensor *vec = input(1/*vector const input*/);
    
    ARM_COMPUTE_ERROR_ON(src == nullptr);
    ARM_COMPUTE_ERROR_ON(vec == nullptr);

    return compute_output_descriptor(src->desc(),vec->desc(), _info);
}


NodeType TokenEmbeddingLayerNode::type() const
{
    return NodeType::TokenEmbeddingLayer;
}

void TokenEmbeddingLayerNode::accept(INodeVisitor &v)
{
    v.visit(*this);
}
} // namespace graph
} // namespace arm_compute
