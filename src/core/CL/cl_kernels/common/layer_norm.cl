#include "helpers.h"

#define MUL_OP(x, y) x *y
#define ADD_OP(x, y) x + y
#define DIV_OP(x, y) x / y
#define POW_OP(x, y) pow(x, y)
#define SQCVT_SAT(a) a

#define sum(in0, in1, size) (in0 + SUM_REDUCE(in1, size))
#define square_sum(in0, in1, size) (in0 + SUM_REDUCE((in1 * in1), size))
#define product(in0, in1, size) (in0 * PROD_REDUCE(in1, size))
#define min_(in0, in1, size) (min(in0, MIN_REDUCE(in1, size)))
#define max_(in0, in1, size) (max(in0, MAX_REDUCE(in1, size)))

/** This kernel performs l2 normalization on x-axis
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE_X=size. e.g. -DVEC_SIZE_X=16
 * @note The leftover size in the X dimension shoud be given as preprocessor argument using -DVEC_SIZE_LEFTOVER_X is; x_dimension % VEC_SIZE_X. e.g. -DVEC_SIZE_LEFTOVER_X=1
 *
 * @param[in]  input_ptr                            Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  epsilon                              Epsilon value
 */
 /*
void layer_norm_fp32(const ITensor *src, ITensor *dst, const Window &window, float epsilon,
                     float gamma,
                     float beta,
                     int   layer_axis)
{
    const int  window_step_axis  = 1;
    const auto window_start_axis = static_cast<int>(window[layer_axis].start());
    const auto window_end_axis   = static_cast<int>(window[layer_axis].end());

    Window win = window; //.collapse_if_possible(window, Window::DimZ)
    win.set(layer_axis, Window::Dimension(0, 1, 1));

    Iterator input(src, win);
    Iterator output(dst, win);

    execute_window_loop(
        win,
        [&](const Coordinates &)
        {
            const auto input_ptr  = reinterpret_cast<const float *>(input.ptr());
            const auto output_ptr = reinterpret_cast<float *>(output.ptr());
            float      mean       = 0;
            float      var        = 0;
            float      res;

            const int axis_len = window_end_axis - window_step_axis;
            // Calculate mean 
            int axis = window_start_axis;
            for(; axis <= axis_len; axis += window_step_axis)
            {
                mean += *(input_ptr + axis);
            }
            mean = mean / (axis_len + 1);

            // Calculate variance 
            axis = window_start_axis;
            for(; axis <= axis_len; axis += window_step_axis)
            {
                var += (*(input_ptr + axis) - mean) * (*(input_ptr + axis) - mean);
            }
            var = var / (axis_len + 1);

            // Calculate layer normalization
            axis = window_start_axis;
            for(; axis <= axis_len; axis += window_step_axis)
            {
                res                                           = ((*(input_ptr + axis) - mean) / sqrt(var + epsilon)) * gamma + beta;
                *reinterpret_cast<float *>(output_ptr + axis) = res;
            }

            ARM_COMPUTE_UNUSED(epsilon);
            ARM_COMPUTE_UNUSED(input_ptr);
            ARM_COMPUTE_UNUSED(output_ptr);
        },
        input, output);
}
*/
__kernel void layer_norm(TENSOR3D_DECLARATION(input),
                         TENSOR3D_DECLARATION(output),
                         DATA_TYPE epsilon,
                         DATA_TYPE gamma,
                         DATA_TYPE beta)
{
    int y = get_global_id(1);
    int z = get_global_id(2);

    __global uchar *input_addr  = input_ptr + input_offset_first_element_in_bytes + y * input_stride_y;
    __global uchar *output_addr = output_ptr + output_offset_first_element_in_bytes + y * output_stride_y;

    DATA_TYPE res = (DATA_TYPE)0;
    DATA_TYPE mean;
    DATA_TYPE var = (DATA_TYPE)0;
    DATA_TYPE sqrt_var_epsilon;

    int x = 0;
    // Calculate mean
    for(; x <= (WIDTH - VEC_SIZE); x += VEC_SIZE)
    {
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE) vals = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(input_addr + x * input_stride_x));
        res  = sum(res, vals, VEC_SIZE);
    }

#if(WIDTH % VEC_SIZE)
    for(; x < WIDTH; ++x)
    {
        DATA_TYPE val = *((__global DATA_TYPE *)(input_addr + x * input_stride_x));
        res           = sum(res, val, 1);
    }

#endif // (WIDTH % VEC_SIZE)

    mean = res / WIDTH;
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE) means = mean;

    x = 0;
    // Calculate variance
    for(; x <= (WIDTH - VEC_SIZE); x += VEC_SIZE)
    {
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE) vals = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(input_addr + x * sizeof(DATA_TYPE)));
        vals  = vals - mean;
        vals  = vals * vals;
        var   = sum(var, vals, VEC_SIZE);
    }

#if(WIDTH % VEC_SIZE)
    for(; x < WIDTH; ++x)
    {
        DATA_TYPE val = *((__global DATA_TYPE *)(input_addr + x * sizeof(DATA_TYPE)));
        val           = val - mean;
        val           = val * val;
        var           = sum(var, val, 1);
    }
#endif // (WIDTH % VEC_SIZE)

    var = var / WIDTH;
    sqrt_var_epsilon = sqrt(var + epsilon);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE) sqrt_var_epsilons = sqrt_var_epsilon;
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE) gammas = gamma;
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE) betas = beta; 

    x = 0;

    // Calculate layer normalization
    for(; x <= (WIDTH - VEC_SIZE); x += VEC_SIZE)
    {
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE) vals = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)(input_addr + x * sizeof(DATA_TYPE)));
        vals = vals - mean;
        vals = vals / sqrt_var_epsilons;
        vals = vals * gammas;
        vals = vals + betas;
        
        VSTORE(VEC_SIZE)(vals, 0, (__global DATA_TYPE *)(output_addr + x * output_stride_x));
    }

#if(WIDTH % VEC_SIZE)
    for(; x < WIDTH; ++x)
    {
        DATA_TYPE val = *((__global DATA_TYPE *)(input_addr + x * sizeof(DATA_TYPE)));
        val = val - mean;
        val = val / sqrt_var_epsilon;
        val = val * gamma;
        val = val + beta;
        VSTORE(1)(val, 0, (__global DATA_TYPE *)(output_addr + x * output_stride_x));
    
    }
#endif // (WIDTH % VEC_SIZE)


}
