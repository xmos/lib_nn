#pragma once

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

#define C_API EXTERN_C

#define ERR_MSG_DESCRIPTOR_FAIL_BYTES() (128)

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif
