// FP32 Mamba-1 single-frame recurrence/readout, retaining caller-owned state.
#include "include/onnxruntime_c_api.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <vector>

struct Kernel { const OrtApi* api; };

// The build uses the baseline ISA, so loading the library is safe on any
// x86-64 host; only this loop is cloned, and the loader's ifunc resolver picks
// the widest variant the running CPU supports. Never build with -march=native:
// a host without that ISA would die of SIGILL inside Run, where no fallback
// can catch it.
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define PURESOUND_ISA_CLONES __attribute__((target_clones("avx512f", "avx2", "default")))
#else
#define PURESOUND_ISA_CLONES
#endif

// Sixteen fp32 lanes. GCC lowers it to one zmm, two ymm or four xmm registers
// per clone, so one source serves every ISA the loader can pick. The helpers
// must inline: called across a clone boundary, a vector argument changes ABI.
typedef float vf16 __attribute__((vector_size(64)));
typedef int32_t vi16 __attribute__((vector_size(64)));
#define LANES_INLINE static inline __attribute__((always_inline))

LANES_INLINE vf16 load16(const float* p) { vf16 v; std::memcpy(&v, p, sizeof v); return v; }
LANES_INLINE void store16(float* p, vf16 v) { std::memcpy(p, &v, sizeof v); }
LANES_INLINE vf16 splat16(float x) { return vf16{} + x; }

// exp on 16 lanes, inline, because a libm call per 16 values costs more than
// the values. Round-to-nearest Cody-Waite reduction and the Cephes degree-6
// polynomial, ~2 ulp. Inputs are clamped so the exponent stays finite.
LANES_INLINE vf16 exp16(vf16 x) {
  x = x < -87.3f ? splat16(-87.3f) : x;
  x = x > 88.3f ? splat16(88.3f) : x;
  const vf16 t = x * 1.44269504088896341f + 0.5f;
  vi16 n = __builtin_convertvector(t, vi16);
  n += (vi16)(__builtin_convertvector(n, vf16) > t);  // truncation -> floor
  const vf16 fn = __builtin_convertvector(n, vf16);
  vf16 r = x - fn * 0.693359375f;
  r = r - fn * -2.12194440e-4f;
  vf16 p = splat16(1.9875691500e-4f);
  p = p * r + 1.3981999507e-3f;
  p = p * r + 8.3334519073e-3f;
  p = p * r + 4.1665795894e-2f;
  p = p * r + 1.6666665459e-1f;
  p = p * r + 5.0000001201e-1f;
  p = p * r * r + r + 1.0f;
  return p * (vf16)((n + 127) << 23);
}

LANES_INLINE float sum16(vf16 v) {
  v += __builtin_shuffle(v, vi16{8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7});
  v += __builtin_shuffle(v, vi16{4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11});
  v += __builtin_shuffle(v, vi16{2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13});
  v += __builtin_shuffle(v, vi16{1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14});
  return v[0];
}

PURESOUND_ISA_CLONES
static void ssm_step(int64_t n, int64_t d, int64_t s, const float* dt,
                     const float* u, const float* b, const float* c,
                     const float* h, const float* a, const float* skip,
                     const float* z, float* next_h, float* y) {
  // No thread pool or per-session mutable state: independent streams can
  // share this library. The SiLU gate is elementwise, so it gets one
  // contiguous pass (y holds it until the readout multiplies in). The clamp
  // keeps exp finite: under -ffast-math the division becomes a reciprocal
  // estimate, which turns z / inf into NaN rather than -0.
  const size_t nd = static_cast<size_t>(n) * d;
  for (size_t i = 0; i < nd; ++i) y[i] = z[i] / (1.0f + std::exp(std::fmin(-z[i], 88.0f)));
  if (s == 16) {
    // Mamba's default state width: each channel's state is exactly one
    // 16-lane vector, so the update, decay and readout need no loop over S.
    for (int64_t batch = 0; batch < n; ++batch) {
      const vf16 bb = load16(b + batch * 16), cc = load16(c + batch * 16);
      for (int64_t channel = 0; channel < d; ++channel) {
        const size_t idx = static_cast<size_t>(batch) * d + channel;
        const float delta = dt[idx];
        const vf16 next = load16(h + idx * 16) * exp16(load16(a + channel * 16) * delta)
                          + bb * (delta * u[idx]);
        store16(next_h + idx * 16, next);
        y[idx] *= sum16(next * cc) + skip[channel] * u[idx];
      }
    }
    return;
  }
  for (int64_t batch = 0; batch < n; ++batch) {
    for (int64_t channel = 0; channel < d; ++channel) {
      const size_t idx = static_cast<size_t>(batch) * d + channel;
      const size_t base = idx * s;
      const float delta = dt[idx], du = delta * u[idx];
      float sum = 0.0f;
      for (int64_t k = 0; k < s; ++k) {
        const float decay = std::exp(delta * a[channel * s + k]);
        const float next = h[base + k] * decay + du * b[batch * s + k];
        next_h[base + k] = next;
        sum += next * c[batch * s + k];
      }
      y[idx] *= sum + skip[channel] * u[idx];
    }
  }
}

static OrtStatus* ORT_API_CALL create(const OrtCustomOp*, const OrtApi* api,
                                    const OrtKernelInfo*, void** kernel) {
  try { *kernel = new Kernel{api}; return nullptr; }
  catch (const std::exception& e) { return api->CreateStatus(ORT_FAIL, e.what()); }
}

static OrtStatus* run(Kernel* kernel, OrtKernelContext* context) {
  const OrtApi* api = kernel->api;
  const float* input[8];
  std::vector<int64_t> shapes[8];
  for (size_t i = 0; i < 8; ++i) {
    const OrtValue* value = nullptr;
    auto* status = api->KernelContext_GetInput(context, i, &value);
    if (status) return status;
    OrtTensorTypeAndShapeInfo* info = nullptr;
    status = api->GetTensorTypeAndShape(value, &info);
    if (status) return status;
    size_t rank = 0;
    ONNXTensorElementDataType dtype;
    status = api->GetTensorElementType(info, &dtype);
    if (!status) status = api->GetDimensionsCount(info, &rank);
    if (status) { api->ReleaseTensorTypeAndShapeInfo(info); return status; }
    if (dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || rank > 3) {
      api->ReleaseTensorTypeAndShapeInfo(info);
      return api->CreateStatus(ORT_INVALID_ARGUMENT, "FusedSsmStep expects FP32 tensors of rank at most three");
    }
    shapes[i].resize(rank);
    status = api->GetDimensions(info, shapes[i].data(), rank);
    api->ReleaseTensorTypeAndShapeInfo(info);
    if (status) return status;
    size_t count = 1;
    for (auto dim : shapes[i]) {
      if (dim <= 0 || static_cast<uint64_t>(dim) >
          std::numeric_limits<size_t>::max() / sizeof(float) / count)
        return api->CreateStatus(ORT_INVALID_ARGUMENT, "FusedSsmStep dimensions must be positive and fit in memory");
      count *= static_cast<size_t>(dim);
    }
    void* ptr = nullptr;
    status = api->GetTensorMutableData(const_cast<OrtValue*>(value), &ptr);
    if (status) return status;
    input[i] = static_cast<const float*>(ptr);
  }
  if (shapes[4].size() != 3)
    return api->CreateStatus(ORT_INVALID_ARGUMENT, "FusedSsmStep state must be [N,D,S]");
  const int64_t n = shapes[4][0], d = shapes[4][1], s = shapes[4][2];
  const std::vector<int64_t> nd{n, d}, ns{n, s}, ds{d, s}, channels{d};
  if (shapes[0] != nd || shapes[1] != nd || shapes[2] != ns ||
      shapes[3] != ns || shapes[5] != ds || shapes[6] != channels || shapes[7] != nd)
    return api->CreateStatus(ORT_INVALID_ARGUMENT, "FusedSsmStep incompatible dt/u/B/C/h/A/D/z shapes");
  OrtValue* values[2];
  auto* status = api->KernelContext_GetOutput(context, 0, shapes[4].data(), 3, &values[0]);
  if (status) return status;
  status = api->KernelContext_GetOutput(context, 1, nd.data(), 2, &values[1]);
  if (status) return status;
  void *h_ptr = nullptr, *y_ptr = nullptr;
  status = api->GetTensorMutableData(values[0], &h_ptr);
  if (status) return status;
  status = api->GetTensorMutableData(values[1], &y_ptr);
  if (status) return status;
  ssm_step(n, d, s, input[0], input[1], input[2], input[3], input[4], input[5],
           input[6], input[7], static_cast<float*>(h_ptr), static_cast<float*>(y_ptr));
  return nullptr;
}

static OrtStatus* ORT_API_CALL compute(void* raw, OrtKernelContext* context) {
  auto* kernel = static_cast<Kernel*>(raw);
  try { return run(kernel, context); }
  catch (const std::exception& e) { return kernel->api->CreateStatus(ORT_FAIL, e.what()); }
  catch (...) { return kernel->api->CreateStatus(ORT_FAIL, "FusedSsmStep failed"); }
}
static const char* ORT_API_CALL name(const OrtCustomOp*) { return "FusedSsmStep"; }
static const char* ORT_API_CALL provider(const OrtCustomOp*) { return "CPUExecutionProvider"; }
static size_t ORT_API_CALL inputs(const OrtCustomOp*) { return 8; }
static size_t ORT_API_CALL outputs(const OrtCustomOp*) { return 2; }
static ONNXTensorElementDataType ORT_API_CALL type(const OrtCustomOp*, size_t) {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}
static OrtCustomOpInputOutputCharacteristic ORT_API_CALL required(const OrtCustomOp*, size_t) {
  return INPUT_OUTPUT_REQUIRED;
}
static OrtMemType ORT_API_CALL memory(const OrtCustomOp*, size_t) { return OrtMemTypeDefault; }
static void ORT_API_CALL destroy(void* kernel) { delete static_cast<Kernel*>(kernel); }
static OrtCustomOp op = []() {
  OrtCustomOp value{};
  value.version = ORT_API_VERSION;
  value.CreateKernelV2 = create;
  value.KernelComputeV2 = compute;
  value.GetName = name;
  value.GetExecutionProviderType = provider;
  value.GetInputTypeCount = inputs;
  value.GetOutputTypeCount = outputs;
  value.GetInputType = type;
  value.GetOutputType = type;
  value.GetInputCharacteristic = required;
  value.GetOutputCharacteristic = required;
  value.GetInputMemoryType = memory;
  value.KernelDestroy = destroy;
  return value;
}();

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,
                                                   const OrtApiBase* base) {
  const OrtApi* api = base->GetApi(ORT_API_VERSION);
  if (!api) return base->GetApi(1)->CreateStatus(ORT_FAIL, "FusedSsmStep needs ORT API 20 or newer");
  try {
    struct Domain {
      const OrtApi* api;
      OrtCustomOpDomain* value = nullptr;
      ~Domain() { if (value) api->ReleaseCustomOpDomain(value); }
    };
    // One domain per loaded library, retained through all session lifetimes.
    // Repeated registrations do not allocate or leak another domain.
    static Domain domain{api};
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    if (!domain.value) {
      auto* status = api->CreateCustomOpDomain("com.puresound", &domain.value);
      if (status) return status;
      status = api->CustomOpDomain_Add(domain.value, &op);
      if (status) {
        api->ReleaseCustomOpDomain(domain.value);
        domain.value = nullptr;
        return status;
      }
    }
    return api->AddCustomOpDomain(options, domain.value);
  } catch (const std::exception& e) { return api->CreateStatus(ORT_FAIL, e.what()); }
}
