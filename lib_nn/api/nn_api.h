#pragma once

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

#define C_API EXTERN_C

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif
