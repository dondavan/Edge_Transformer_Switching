#ifndef ARM_COMPUTE_GRAPH_ATTENTION_LINEAR_NODE_H
#define ARM_COMPUTE_GRAPH_ATTENTION_LINEAR_NODE_H

#include "arm_compute/core/Types.h"
#include "arm_compute/graph/INode.h"

namespace arm_compute
{
namespace graph
{
/** Linear Layer node */
class AttentionLinearNode final : public INode
{
    public:
    /** Constructor
     *
     * @param[in] linear_info Contains information described in @ref LinearLayerLayerInfo.
     */
    AttentionLinearNode(LinearLayerInfo linear_info);
    /** Prevent instances of this class from being copy constructed */
    AttentionLinearNode(const AttentionLinearNode &) = delete;
    /** Prevent instances of this class from being copied */
    AttentionLinearNode &operator=(const AttentionLinearNode &) = delete;

    /** LinearLayerInfo accessor
     *
     * @return LinearLayerInfo
     */
    const LinearLayerInfo &linear_info() const;

    // Inherited overridden methods:
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void             accept(INodeVisitor &v) override;

    private:
    LinearLayerInfo _linear_info;
};
} // namespace graph
} // namespace arm_compute

#endif /* ARM_COMPUTE_GRAPH_ATTENTION_LINEAR_NODE_H */
