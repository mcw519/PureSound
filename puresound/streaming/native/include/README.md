`onnxruntime_c_api.h` is the unmodified ONNX Runtime v1.20.0 C API header:
https://github.com/microsoft/onnxruntime/blob/v1.20.0/include/onnxruntime/core/session/onnxruntime_c_api.h

Copyright Microsoft Corporation; distributed under the adjacent MIT LICENSE.
API 20 keeps the optional custom library compatible with supported newer ORT
releases. The library uses ORT's supplied API table and does not link against a
separate ORT installation. This header is only needed by the local build command.
