/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/graph.h"
#ifdef ARM_COMPUTE_CL
#include "arm_compute/runtime/CL/Utils.h"
#endif /* ARM_COMPUTE_CL */
#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

using namespace arm_compute;
using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

class GraphGPTExample : public Example
{
    public:
    GraphGPTExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "GPT")
    {}

    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);
        cmd_parser.validate();

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);

        // Return when help menu is requested
        if(common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        // Print parameter values
        std::cout << common_params << std::endl;

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        constexpr unsigned int d_model    = 768U;   // Dim layer output
        constexpr unsigned int d_vocab    = 50257U; // Vocabulary size
        constexpr unsigned int d_segemnt  = 1U;     // no segmentation in gpt2
        constexpr unsigned int d_position = 1024U;   // Pretrained positional encoding length
        constexpr unsigned int h          = 12U;    // Parallel attention (Heads)
        constexpr float        eps        = 1e-5;  // Layer normalization eplision
        constexpr unsigned int d_ff       = 3072U;  // Dim feedforward

        // Create input tensor
        const TensorShape src_tensor = TensorShape(common_params.input_len);

        // Data layout
        const DataLayout operation_layout = DataLayout::NCHW;

        TensorDescriptor input_descriptor = TensorDescriptor(src_tensor, common_params.data_type);

        // Set graph hints
        graph << common_params.target << common_params.fast_math_hint;

        // Text preprocessor
        std::unique_ptr<IPreprocessor> at2_preproccessor = std::make_unique<atoiPreprocessor>();
        // Encode Input
        // RULE: segment id must all be the same and the segment embedding parameters are all 0
        graph << InputLayer(input_descriptor, get_token_accessor(common_params),
                            get_segment_accessor(common_params.segment, move(at2_preproccessor)))
                     .set_name("in1")

            << EmbeddingLayer(EmbeddingLayerInfo(d_model,
                                                   d_vocab,
                                                   d_segemnt,
                                                   d_position,
                                                   true /*Use pretrained positional encoding*/,
                                                   ConvertPolicy::SATURATE),
                                get_weights_accessor(data_path, "token_embedding.npy", operation_layout),
                                // all zeroes for gpt2
                                get_weights_accessor(data_path, "segment_embedding.npy", operation_layout),
                                get_weights_accessor(data_path, "position_embedding.npy", operation_layout))
                     .set_name("tkemb1");

        add_decoder_block(data_path, "layer_0/" /*Layer Parameter Dir*/, d_model, h, eps, d_ff);
        add_decoder_block(data_path, "layer_1/" /*Layer Parameter Dir*/, d_model, h, eps, d_ff);
        add_decoder_block(data_path, "layer_2/" /*Layer Parameter Dir*/, d_model, h, eps, d_ff);
        add_decoder_block(data_path, "layer_3/" /*Layer Parameter Dir*/, d_model, h, eps, d_ff);
        add_decoder_block(data_path, "layer_4/" /*Layer Parameter Dir*/, d_model, h, eps, d_ff);
        add_decoder_block(data_path, "layer_5/" /*Layer Parameter Dir*/, d_model, h, eps, d_ff);
        add_decoder_block(data_path, "layer_6/" /*Layer Parameter Dir*/, d_model, h, eps, d_ff);
        add_decoder_block(data_path, "layer_7/" /*Layer Parameter Dir*/, d_model, h, eps, d_ff);
        add_decoder_block(data_path, "layer_8/" /*Layer Parameter Dir*/, d_model, h, eps, d_ff);
        add_decoder_block(data_path, "layer_9/" /*Layer Parameter Dir*/, d_model, h, eps, d_ff);
        add_decoder_block(data_path, "layer_10/" /*Layer Parameter Dir*/, d_model, h, eps, d_ff);
        add_decoder_block(data_path, "layer_11/" /*Layer Parameter Dir*/, d_model, h, eps, d_ff);

        graph << LayerNormLayer(LayerNormLayerInfo(0 /*Window::DimX*/, eps))
            // TODO: get correct dimensions and parameters
            << LinearLayer(LinearLayerInfo(d_model, TensorShape(d_model, d_vocab),
                                            TensorShape(d_vocab), 1),
                             get_weights_accessor(data_path, "projection_weight.npy"),
                             // just zeroes for gpt2
                             get_weights_accessor(data_path, "projection_bias.npy"))

              << OutputLayer(get_output_accessor(common_params)).set_name("out1");
        
        // Finalize graph
        GraphConfig config;

        config.num_threads = common_params.threads;
        config.use_tuner   = common_params.enable_tuner;
        config.tuner_mode  = common_params.tuner_mode;
        config.tuner_file  = common_params.tuner_file;
        config.mlgo_file   = common_params.mlgo_file;

        config.use_transition_memory_manager = false;

#ifdef MEASURE_TIME
        // Clear previous output
        std::ofstream ofs;
        ofs.open("measure_output.txt", std::ofstream::out | std::ofstream::trunc);
        ofs.close();
#endif

        graph.finalize(common_params.target, config);

        return true;
    }

    void do_run() override
    {
        graph.run();
    }

    private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;

    void add_decoder_block(std::string data_path, std::string layer_path,
                           unsigned int d_model, unsigned int h, float eps, unsigned int d_ff)
    {
        SubStream with_attention(graph);
        SubStream without_attention(graph);

        with_attention << LayerNormLayer(LayerNormLayerInfo(0 /*Window::DimX*/, eps));

        // TODO: multihead MASKED attention
        with_attention << AttentionLinearLayer(LinearLayerInfo(d_model),
                                    get_weights_accessor(data_path + layer_path, "query_weight.npy"),
                                    get_weights_accessor(data_path + layer_path, "query_bias.npy"),
                                    get_weights_accessor(data_path + layer_path, "key_weight.npy"),
                                    get_weights_accessor(data_path + layer_path, "key_bias.npy"),
                                    get_weights_accessor(data_path + layer_path, "value_weight.npy"),
                                    get_weights_accessor(data_path + layer_path, "value_bias.npy"))
            << ScaleDotProductionLayer(ScaleDotProductionLayerInfo(d_model, h)).set_name("mha1");

        // add and norm
        graph << EltwiseLayer(std::move(with_attention), std::move(without_attention), EltwiseOperation::Add).set_name("add_4_norm_attention")
            << LayerNormLayer(LayerNormLayerInfo(0 /*Window::DimX*/, eps));

        SubStream without_ff(graph);
        SubStream with_ff(graph);

        with_ff << LinearLayer(LinearLayerInfo(d_ff, TensorShape(d_model, d_ff) /*weight*/,
                                                        TensorShape(d_ff) /*bias*/),
                               get_weights_accessor(data_path + layer_path, "ff_weight_0.npy"),
                               get_weights_accessor(data_path + layer_path, "ff_bias_0.npy"))
                << ActivationLayer(ActivationLayerInfo(ActivationFunction::GELU))
                << LinearLayer(LinearLayerInfo(d_model, TensorShape(d_ff, d_model) /*weight*/,
                                               TensorShape(d_model) /*bias*/),
                               get_weights_accessor(data_path + layer_path, "ff_weight_1.npy"),
                               get_weights_accessor(data_path + layer_path, "ff_bias_1.npy"));

        graph << EltwiseLayer(std::move(with_ff), std::move(without_ff), EltwiseOperation::Add).set_name("add_4_norm_ff");
    }
};

int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphGPTExample>(argc, argv);
}
