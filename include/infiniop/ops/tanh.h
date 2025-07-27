#ifndef __INFINIOP_TANH_API_H__
#define __INFINIOP_TANH_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopTanhDescriptor_t;

__C __export infiniStatus_t infiniopCreateTanhDescriptor(infiniopHandle_t handle,
                                                        infiniopTanhDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t y,
                                                        infiniopTensorDescriptor_t x);

__C __export infiniStatus_t infiniopGetTanhWorkspaceSize(infiniopTanhDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopTanh(infiniopTanhDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *y,
                                        const void *x,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroyTanhDescriptor(infiniopTanhDescriptor_t desc);

#endif
