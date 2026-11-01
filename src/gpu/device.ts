export interface GpuContext {
  adapter: GPUAdapter
  device: GPUDevice
}

// Some Dawn builds tear down the instance when the adapter wrapper is garbage-collected, which
// destroys the device mid-run; keeping a module-level reference avoids that.
let retainedAdapter: GPUAdapter | null = null

/** The adapter obtained by the last `requestGpu` call (kept alive, see above). */
export const getRetainedAdapter = () => retainedAdapter

export class WebGpuUnavailableError extends Error {}

export async function requestGpu(): Promise<GpuContext> {
  if (!('gpu' in navigator)) {
    throw new WebGpuUnavailableError(`WebGPU is not available (secure context: ${window.isSecureContext}). On Linux it must be enabled manually: Chrome needs --enable-unsafe-webgpu --enable-features=Vulkan; Firefox needs dom.webgpu.enabled in about:config.`)
  }
  // Some Linux setups (integrated GPU only) return null when a power preference is given.
  const adapter =
    (await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })) ??
    (await navigator.gpu.requestAdapter())
  if (!adapter) throw new WebGpuUnavailableError('WebGPU exists but no adapter was returned. On Linux, Chrome usually needs --enable-features=Vulkan (and --ozone-platform=x11 may help); check chrome://gpu.')
  retainedAdapter = adapter
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  })
  return { adapter, device }
}
