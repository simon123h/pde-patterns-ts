export interface GpuContext {
  adapter: GPUAdapter
  device: GPUDevice
}

export class WebGpuUnavailableError extends Error {}

export async function requestGpu(): Promise<GpuContext> {
  if (!('gpu' in navigator)) {
    throw new WebGpuUnavailableError(`WebGPU is not available (secure context: ${window.isSecureContext}). On Linux it must be enabled manually: Chrome needs --enable-unsafe-webgpu --enable-features=Vulkan; Firefox needs dom.webgpu.enabled in about:config.`)
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) throw new WebGpuUnavailableError('WebGPU exists but no adapter was returned. On Linux, Chrome usually needs --enable-features=Vulkan (and --ozone-platform=x11 may help); check chrome://gpu.')
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  })
  return { adapter, device }
}
