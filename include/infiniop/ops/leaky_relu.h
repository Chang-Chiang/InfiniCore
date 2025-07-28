#ifndef __INFINIOP_LEAKY_RELU_API_H__
#define __INFINIOP_LEAKY_RELU_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLeakyReluDescriptor_t;

__C __export infiniStatus_t infiniopCreateLeakyReluDescriptor(infiniopHandle_t handle,
                                                              infiniopLeakyReluDescriptor_t *desc_ptr,
                                                              infiniopTensorDescriptor_t y,
                                                              infiniopTensorDescriptor_t x,
                                                              infiniopTensorDescriptor_t negative_slope);

__C __export infiniStatus_t infiniopGetLeakyReluWorkspaceSize(infiniopLeakyReluDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopLeakyRelu(infiniopLeakyReluDescriptor_t desc,
                                         void *workspace,
                                         size_t workspace_size,
                                         void *y,
                                         const void *x,
                                         const void *negative_slope,
                                         void *stream);

__C __export infiniStatus_t infiniopDestroyLeakyReluDescriptor(infiniopLeakyReluDescriptor_t desc);

#endif
