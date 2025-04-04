#ifndef ARM_COMPUTE_GRAPH_SCALE_DOT_PRODUCTION_ATTENTION_NODE_H
#define ARM_COMPUTE_GRAPH_SCALE_DOT_PRODUCTION_ATTENTION_NODE_H

#include "arm_compute/core/Types.h"
#include "arm_compute/graph/INode.h"

namespace arm_compute
{
namespace graph
{
/** Scale Dot Production Attention node */
class ScaleDotProductionAttentionNode final : public INode
{
public:
    /** Constructor
     *
     * @param[in] sdpa_info Contains information described in @ref ScaleDotProductionLayerInfo.
     */
    ScaleDotProductionAttentionNode(ScaleDotProductionLayerInfo sdpa_info);
    /** Prevent instances of this class from being copy constructed */
    ScaleDotProductionAttentionNode(const ScaleDotProductionAttentionNode &) = delete;
    /** Prevent instances of this class from being copied */
    ScaleDotProductionAttentionNode &operator=(const ScaleDotProductionAttentionNode &) = delete;

    /** ScaleDotProductionLayerInfo accessor
     *
     * @return ScaleDotProductionLayerInfo
     */
    const ScaleDotProductionLayerInfo &sdpa_info() const;

    // Inherited overridden methods:
    NodeType         type() const override;
    bool             forward_descriptors() override;
    TensorDescriptor configure_output(size_t idx) const override;
    void             accept(INodeVisitor &v) override;

private:
    ScaleDotProductionLayerInfo _sdpa_info;
};
} // namespace graph
} // namespace arm_compute

#endif /* ARM_COMPUTE_GRAPH_SCALE_DOT_PRODUCTION_ATTENTION_NODE_H */
