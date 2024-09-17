#include "helpers.h"

/** Perform token vectorization
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 *
 * @param[in]  src_ptr                              Pointer to the first source tensor. Supported data types: All
 * @param[in]  src_stride_x                         Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                           input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                         Stride of the first source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                           input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                         Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                           input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes    The offset of the first element in the first source tensor
 * @param[in]  vector_ptr                            Pointer to the first source tensor. Supported data types: All
 * @param[in]  vector_stride_x                       Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  vector_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  vector_stride_y                       Stride of the first source tensor in Y dimension (in bytes)
 * @param[in]  vector_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  vector_stride_z                       Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  vector_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  vector_offset_first_element_in_bytes  The offset of the first element in the first source tensor
 * @param[out] output_ptr                            Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                       Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                         output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                       Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                         output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                         output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes  The offset of the first element in the destination tensor
 */
__kernel void vectorize(TENSOR3D_DECLARATION(src),
                        TENSOR3D_DECLARATION(vector),
                        TENSOR3D_DECLARATION(output))
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);

    // Compute the output linearized index
    int out_linear_idx = id_y * output_stride_y + id_x * output_stride_x;

    // Compute the src linearized index
    int src_linear_idx = id_y * src_stride_x;

    src_ptr += src_offset_first_element_in_bytes + src_linear_idx;
    // Compute the vector linearized index
    uint vector_offset = *((__global unsigned int *)src_ptr);
    int vector_linear_idx = vector_offset * vector_stride_y + id_x * vector_stride_x;

    // Store result
    output_ptr += output_offset_first_element_in_bytes + out_linear_idx;
    vector_ptr += vector_offset_first_element_in_bytes + vector_linear_idx;
    *((__global DATA_TYPE_DST *)output_ptr) = *((__global DATA_TYPE_VEC *)vector_ptr);
}
