/*
 * Copyright (c) 2016-2023 Arm Limited.
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
#ifndef ACL_ARM_COMPUTE_FUNCTION_INFO_SDPAINFO
#define ACL_ARM_COMPUTE_FUNCTION_INFO_SDPAINFO

#include "arm_compute/core/CoreTypes.h"
#include "arm_compute/core/ITensor.h"
namespace arm_compute
{
struct recurrence_object
{
    unsigned int recurrence_count;
    ITensor     *query;
    ITensor     *key;
    ITensor     *value;
    ITensor     *output;
};

/** Scale Dot Production Attention Layer Information Class*/
class ScaleDotProductionLayerInfo final
{
    public:
    /** Constructor
     *
     * @param[in] d_model   Model dimesion
     * @param[in] h         Parallel attention dimesion
     */
    ScaleDotProductionLayerInfo(unsigned int d_model = 512, unsigned int h = 8)
        : _d_model(d_model),
          _h(h)

    {
        _sdpa_recurrence.recurrence_count = 0;
        _sdpa_recurrence.query            = nullptr;
        _sdpa_recurrence.key              = nullptr;
        _sdpa_recurrence.value            = nullptr;
        _sdpa_recurrence.output           = nullptr;
    }

    /** Constructor using Multi-head attention layer info
     *
     * @param[in] mha_info   MultiHeadAttentionLayerInfo
     */
    ScaleDotProductionLayerInfo(MultiHeadAttentionLayerInfo mha_info)
        : _d_model(mha_info.d_model()),
          _h(mha_info.h())
    {
    }

    /* Get Model dimesion */
    unsigned int d_model() const
    {
        return _d_model;
    }

    /* Get Parallel attention dimesion */
    unsigned int h() const
    {
        return _h;
    }

    struct recurrence_object sdpa_recurrence()
    {
        return _sdpa_recurrence;
    }

    private:
    unsigned int             _d_model;
    unsigned int             _h;
    struct recurrence_object _sdpa_recurrence;
};

} // namespace arm_compute
#endif /* ACL_ARM_COMPUTE_FUNCTION_INFO_SDPAINFO */
