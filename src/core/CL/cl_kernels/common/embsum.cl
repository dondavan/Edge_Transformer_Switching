#include "helpers.h"

/** Perform token segemntization
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 *
 * @param[in]  token_ptr                              Pointer to the first source tensor. Supported data types: All
 * @param[in]  token_stride_x                         Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  token_step_x                           input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  token_stride_y                         Stride of the first source tensor in Y dimension (in bytes)
 * @param[in]  token_step_y                           input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  token_stride_z                         Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  token_step_z                           input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  token_offset_first_element_in_bytes    The offset of the first element in the first source tensor
 * @param[in]  segemnt_ptr                            Pointer to the first source tensor. Supported data types: All
 * @param[in]  segemnt_stride_x                       Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  segemnt_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  segemnt_stride_y                       Stride of the first source tensor in Y dimension (in bytes)
 * @param[in]  segemnt_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  segemnt_stride_z                       Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  segemnt_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  segemnt_offset_first_element_in_bytes  The offset of the first element in the first source tensor
 * @param[in]  position_ptr                            Pointer to the first source tensor. Supported data types: All
 * @param[in]  position_stride_x                       Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  position_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  position_stride_y                       Stride of the first source tensor in Y dimension (in bytes)
 * @param[in]  position_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  position_stride_z                       Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  position_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  position_offset_first_element_in_bytes  The offset of the first element in the first source tensor
 * @param[out] output_ptr                            Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                       Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                         output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                       Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                         output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                         output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes  The offset of the first element in the destination tensor
 */
__kernel void embsum(TENSOR3D_DECLARATION(token),
                     TENSOR3D_DECLARATION(segemnt),
                     TENSOR3D_DECLARATION(position),
                     TENSOR3D_DECLARATION(output))
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);

    // Compute the output linearized index
    int linear_idx = id_y * output_stride_y + id_x * output_stride_x;

    // Store result
    token_ptr       += token_offset_first_element_in_bytes + linear_idx;
    segemnt_ptr     += segemnt_offset_first_element_in_bytes + linear_idx;
    position_ptr    += position_offset_first_element_in_bytes + linear_idx;
    output_ptr      += output_offset_first_element_in_bytes + linear_idx;

    *((__global DATA_TYPE *)output_ptr) = *((__global DATA_TYPE *)token_ptr ) + 
                                          *((__global DATA_TYPE *)segemnt_ptr ) + 
                                          *((__global DATA_TYPE *)position_ptr ) ;
}
