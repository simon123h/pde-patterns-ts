export interface GpuContext {
  adapter: GPUAdapter
  device: GPUDevice
}

export class WebGpuUnavailableError extends Error {}

export async function requestGpu(): Promise<GpuContext> {
  if (!('gpu' in navigator)) {
    throw new WebGpuUnavailableError('WebGPU is not supported by this browser.')
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) throw new WebGpuUnavailableError('No suitable WebGPU adapter found.')
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  })
  return { adapter, device }
}
