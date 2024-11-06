

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

class GraphVanillaTransformerExample : public Example
{
    public:
    GraphVanillaTransformerExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "Vanilla_Transformer")
    {
    }
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

        // Model parameters
        constexpr unsigned int d_model    = 768U;   // Dim layer output
        constexpr unsigned int d_vocab    = 30522U; // Vocaboary size
        constexpr unsigned int d_segemnt  = 2U;     // Sentence segmentation size
        constexpr unsigned int d_position = 512U;   // Pretrained positional encoding length
        constexpr unsigned int h          = 12U;    // Parallel attention (Heads)
        constexpr float        eps        = 1e-12;  // Layer normalization eplision
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
        graph << InputLayer(input_descriptor, get_token_accessor(common_params),
                            get_segment_accessor(common_params.segment, move(at2_preproccessor)))
                     .set_name("in").set_target(Target::NEON)

              << EmbeddingLayer(EmbeddingLayerInfo(d_model,
                                                   d_vocab,
                                                   d_segemnt,
                                                   d_position,
                                                   true /*Use pretrained positional encoding*/,
                                                   ConvertPolicy::SATURATE),
                                get_weights_accessor(data_path, "token_embedding.npy", operation_layout),
                                get_weights_accessor(data_path, "segment_embedding.npy", operation_layout),
                                get_weights_accessor(data_path, "positional_embedding.npy", operation_layout))
                     .set_name("tkemb").set_target(Target::NEON);

        
        add_encoder_block(data_path, "layer_0/" /*Layer Parameter Dir*/, d_model, h, eps, d_ff);
        add_encoder_block(data_path, "layer_1/" /*Layer Parameter Dir*/, d_model, h, eps, d_ff);
        add_encoder_block(data_path, "layer_2/" /*Layer Parameter Dir*/, d_model, h, eps, d_ff);
        add_encoder_block(data_path, "layer_3/" /*Layer Parameter Dir*/, d_model, h, eps, d_ff);
        add_encoder_block(data_path, "layer_4/" /*Layer Parameter Dir*/, d_model, h, eps, d_ff);
        add_encoder_block(data_path, "layer_5/" /*Layer Parameter Dir*/, d_model, h, eps, d_ff);

        add_encoder_block(data_path, "layer_6/" /*Layer Parameter Dir*/, d_model, h, eps, d_ff);
        add_encoder_block(data_path, "layer_7/" /*Layer Parameter Dir*/, d_model, h, eps, d_ff);
        add_encoder_block(data_path, "layer_8/" /*Layer Parameter Dir*/, d_model, h, eps, d_ff);
        add_encoder_block(data_path, "layer_9/" /*Layer Parameter Dir*/, d_model, h, eps, d_ff);
        add_encoder_block(data_path, "layer_10/" /*Layer Parameter Dir*/, d_model, h, eps, d_ff);
        add_encoder_block(data_path, "layer_11/" /*Layer Parameter Dir*/, d_model, h, eps, d_ff);

        // Pooler
        graph << OutputLayer(get_output_accessor(common_params)).set_name("out").set_target(Target::NEON);

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
        auto start_time = std::chrono::high_resolution_clock::now();

        // Run graph
        graph.run();
        graph.run();
        graph.run();
        graph.run();
        graph.run();

        auto   end_time  = std::chrono::high_resolution_clock::now();
        double cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
        std::cout << "Run cost: " << cost_time << std::endl;
    }

    private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;

    void add_encoder_block(std::string data_path, std::string layer_path,
                           unsigned int d_model, unsigned int h, float eps, unsigned int d_ff)
    {
        ARM_COMPUTE_UNUSED(h,d_model,eps,d_ff,data_path,layer_path);
        SubStream without_attention(graph);
        SubStream with_attention(graph);

        with_attention
            /* Self Attention */
            << AttentionConvLayer(1U, 1U, 1U, 
                                    get_weights_accessor(data_path + layer_path, "query_weight.npy"),
                                    get_weights_accessor(data_path + layer_path, "query_bias.npy"),
                                    get_weights_accessor(data_path + layer_path, "key_weight.npy"),
                                    get_weights_accessor(data_path + layer_path, "key_bias.npy"),
                                    get_weights_accessor(data_path + layer_path, "value_weight.npy"),
                                    get_weights_accessor(data_path + layer_path, "value_bias.npy"),
                                    PadStrideInfo(1,1,0,0)).set_target(Target::CL).set_name("attention_conv")
            << ScaleDotProductionLayer(ScaleDotProductionLayerInfo(d_model, h)).set_name("mha").set_target(Target::NEON);

        graph << EltwiseLayer(std::move(with_attention), std::move(without_attention), EltwiseOperation::Add).set_name("attention_res_add").set_target(Target::NEON);

        /* Self output */
        graph << LayerNormLayer(LayerNormLayerInfo(0 /*Window::DimX*/, eps)).set_target(Target::NEON).set_name("attention_norm");

        SubStream without_ff(graph);
        SubStream with_ff(graph);
        /* Self Intermediate(Feed Forward)*/
        with_ff << ConvolutionLayer(1U, 1U, 1U,
                               get_weights_accessor(data_path + layer_path, "ff_weight_0.npy"),
                               get_weights_accessor(data_path + layer_path, "ff_bias_0.npy"),
                                PadStrideInfo(1, 1, 0, 0)).set_target(Target::NEON).set_name("ff_linear_1")
                << ActivationLayer(ActivationLayerInfo(ActivationFunction::GELU)).set_target(Target::NEON).set_name("ff_acti")
                << ConvolutionLayer(1U, 1U, 1U,
                               get_weights_accessor(data_path + layer_path, "ff_weight_1.npy"),
                               get_weights_accessor(data_path + layer_path, "ff_bias_1.npy"),
                                PadStrideInfo(1, 1, 0, 0)).set_target(Target::NEON).set_name("ff_linear_2");

        graph << EltwiseLayer(std::move(with_ff), std::move(without_ff), EltwiseOperation::Add).set_name("ff_res_add").set_target(Target::NEON);

        /* Output*/
        graph << LayerNormLayer(LayerNormLayerInfo(0 /*Window::DimX*/, eps)).set_target(Target::NEON).set_name("ff_norm");

        
    }
};

/** Main program for Vanilla Transformer
 *
 * Model is based on:
 *      "Attention Is All You Need". 
 *      Ashish Vaswani, Noam Shazeer,Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser,Illia Polosukhin. 2017.
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphVanillaTransformerExample>(argc, argv);
}