/** A 2D transform between physical and spectral space for a batch of fields. */
export interface SpectralTransform {
  /** Spectral modes per field (the spectrum is transposed: index = kx · ny + ky). */
  readonly modes: number
  forward(encoder: GPUCommandEncoder, src: GPUBuffer, dst: GPUBuffer): void
  inverse(encoder: GPUCommandEncoder, src: GPUBuffer, dst: GPUBuffer): void
  destroy(): void
}
