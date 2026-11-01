import rowFftWgsl from './shaders/rowFft.wgsl?raw'
import transposeWgsl from './shaders/transpose.wgsl?raw'

const MAX_ROW_LENGTH = 2048 // 2048 vec2<f32> = 16 KiB, the default workgroup-storage limit

interface Pass {
  pipeline: GPUComputePipeline
  bindGroup: GPUBindGroup
  workgroups: [number, number]
}

/**
 * Complex 2D FFT on the GPU for nx × ny fields (power-of-two sizes, f32, interleaved re/im).
 *
 * Data layout (index = row * rowLength + col):
 *  - physical space: row-major, index = y * nx + x
 *  - spectral space: TRANSPOSED, index = kx * ny + ky
 *
 * Leaving the spectrum transposed saves two transposes per step; pointwise operations in
 * Fourier space only need to map the index back to (kx, ky) accordingly.
 *
 * `inverse` includes the 1/(nx·ny) normalization. `src` and `dst` must be different buffers
 * (both need STORAGE usage); `src` is left untouched.
 */
export class Fft2D {
  private readonly rowPipeline: GPUComputePipeline
  private readonly transposePipeline: GPUComputePipeline
  private readonly scratch: GPUBuffer
  private readonly owned: GPUBuffer[] = []
  private readonly cache = new WeakMap<GPUBuffer, WeakMap<GPUBuffer, Pass[]>>()
  private readonly twX: GPUBuffer
  private readonly twY: GPUBuffer

  constructor(
    private readonly device: GPUDevice,
    readonly nx: number,
    readonly ny: number,
  ) {
    const maxLen = Math.min(MAX_ROW_LENGTH, device.limits.maxComputeWorkgroupStorageSize / 8)
    for (const n of [nx, ny]) {
      if (n < 2 || (n & (n - 1)) !== 0) throw new Error(`FFT size ${n} must be a power of two ≥ 2`)
      if (n > maxLen) throw new Error(`FFT size ${n} exceeds the supported maximum of ${maxLen}`)
    }
    this.rowPipeline = this.pipeline(rowFftWgsl)
    this.transposePipeline = this.pipeline(transposeWgsl)
    this.scratch = this.track(
      device.createBuffer({ size: nx * ny * 8, usage: GPUBufferUsage.STORAGE }),
    )
    this.twX = this.twiddles(nx)
    this.twY = nx === ny ? this.twX : this.twiddles(ny)
  }

  /** Physical (index y * nx + x) → spectral (transposed, index kx * ny + ky). */
  forward(encoder: GPUCommandEncoder, src: GPUBuffer, dst: GPUBuffer): void {
    this.run(encoder, src, dst, 'forward')
  }

  /** Spectral (transposed) → physical, normalized. */
  inverse(encoder: GPUCommandEncoder, src: GPUBuffer, dst: GPUBuffer): void {
    this.run(encoder, src, dst, 'inverse')
  }

  destroy(): void {
    for (const b of this.owned) b.destroy()
  }

  private run(encoder: GPUCommandEncoder, src: GPUBuffer, dst: GPUBuffer, dir: 'forward' | 'inverse') {
    if (src === dst) throw new Error('Fft2D requires distinct src and dst buffers')
    const key = dir === 'forward' ? 0 : 1
    let byDst = this.cache.get(src)
    if (!byDst) this.cache.set(src, (byDst = new WeakMap()))
    let passes = byDst.get(dst)
    if (!passes) byDst.set(dst, (passes = [...this.build(src, dst, 'forward'), ...this.build(src, dst, 'inverse')]))

    const pass = encoder.beginComputePass()
    for (const p of passes.slice(key * 3, key * 3 + 3)) {
      pass.setPipeline(p.pipeline)
      pass.setBindGroup(0, p.bindGroup)
      pass.dispatchWorkgroups(p.workgroups[0], p.workgroups[1])
    }
    pass.end()
  }

  /**
   * Three passes: row FFT (src → dst), transpose (dst → scratch), row FFT (scratch → dst).
   * Forward: rows of length nx first, then ny. Inverse: the reverse order.
   */
  private build(src: GPUBuffer, dst: GPUBuffer, dir: 'forward' | 'inverse'): Pass[] {
    const { nx, ny } = this
    const fwd = dir === 'forward'
    const inv = fwd ? 0 : 1
    // first stage works on rows of length n1 (count r1), second on length n2 (count r2)
    const [n1, r1, tw1, n2, r2, tw2] = fwd ? [nx, ny, this.twX, ny, nx, this.twY] : [ny, nx, this.twY, nx, ny, this.twX]
    const norm = fwd ? 1 : 1 / (nx * ny)

    const rowParams = (n: number, rows: number, scale: number) =>
      this.uniform((v) => {
        v.setUint32(0, n, true)
        v.setUint32(4, Math.log2(n), true)
        v.setUint32(8, rows, true)
        v.setUint32(12, inv, true)
        v.setFloat32(16, scale, true)
      })
    const transposeParams = this.uniform((v) => {
      v.setUint32(0, r1, true) // src rows
      v.setUint32(4, n1, true) // src cols
    })

    const bind = (pipeline: GPUComputePipeline, entries: GPUBuffer[]) =>
      this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: entries.map((buffer, binding) => ({ binding, resource: { buffer } })),
      })

    return [
      { pipeline: this.rowPipeline, bindGroup: bind(this.rowPipeline, [rowParams(n1, r1, 1), src, dst, tw1]), workgroups: [r1, 1] },
      {
        pipeline: this.transposePipeline,
        bindGroup: bind(this.transposePipeline, [transposeParams, dst, this.scratch]),
        workgroups: [Math.ceil(n1 / 16), Math.ceil(r1 / 16)],
      },
      { pipeline: this.rowPipeline, bindGroup: bind(this.rowPipeline, [rowParams(n2, r2, norm), this.scratch, dst, tw2]), workgroups: [r2, 1] },
    ]
  }

  private pipeline(code: string): GPUComputePipeline {
    return this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: this.device.createShaderModule({ code }), entryPoint: 'main' },
    })
  }

  private track(b: GPUBuffer): GPUBuffer {
    this.owned.push(b)
    return b
  }

  private uniform(write: (view: DataView) => void): GPUBuffer {
    const data = new ArrayBuffer(32)
    write(new DataView(data))
    const buffer = this.track(
      this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    )
    this.device.queue.writeBuffer(buffer, 0, data)
    return buffer
  }

  /** tw[k] = exp(−2πi k / n) for k < n/2, computed in f64. */
  private twiddles(n: number): GPUBuffer {
    const table = new Float32Array(n) // n/2 complex entries
    for (let k = 0; k < n / 2; k++) {
      table[2 * k] = Math.cos((2 * Math.PI * k) / n)
      table[2 * k + 1] = -Math.sin((2 * Math.PI * k) / n)
    }
    const buffer = this.track(
      this.device.createBuffer({ size: Math.max(16, table.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
    )
    this.device.queue.writeBuffer(buffer, 0, table)
    return buffer
  }
}
