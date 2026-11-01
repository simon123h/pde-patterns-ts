import rowC2rWgsl from './shaders/rowC2r.wgsl?raw'
import rowFftWgsl from './shaders/rowFft.wgsl?raw'
import rowR2cWgsl from './shaders/rowR2c.wgsl?raw'
import transposeWgsl from './shaders/transpose.wgsl?raw'
import type { SpectralTransform } from './transform'

const MAX_HALF_LENGTH = 2048 // 2048 vec2<f32> = 16 KiB of workgroup memory

interface Pass {
  pipeline: GPUComputePipeline
  bindGroup: GPUBindGroup
  workgroups: [number, number, number]
}

/**
 * Real-to-complex 2D FFT for `batch` real nx × ny fields (f32 in, complex out).
 *
 * Physical space: plain f32 array, index = field·nx·ny + y·nx + x.
 * Spectral space: the non-redundant half kx = 0 … nx/2 of the Hermitian spectrum, complex and
 * transposed: index = field·modes + kx·ny + ky with modes = (nx/2 + 1)·ny.
 *
 * Forward: real rows → half rows (packed n/2-point FFT) → transpose → complex FFT along y.
 * Inverse reverses this and includes the 1/(nx·ny) normalization. `src` is not modified.
 */
export class RealFft2D implements SpectralTransform {
  readonly modes: number
  private readonly halfWidth: number
  private readonly r2c: GPUComputePipeline
  private readonly c2r: GPUComputePipeline
  private readonly rowFft: GPUComputePipeline
  private readonly transpose: GPUComputePipeline
  private readonly scratchA: GPUBuffer
  private readonly scratchB: GPUBuffer
  private readonly owned: GPUBuffer[] = []
  private readonly cache = new WeakMap<GPUBuffer, WeakMap<GPUBuffer, { forward: Pass[]; inverse: Pass[] }>>()
  private readonly twHalf: GPUBuffer // for the n/2-point FFT along x
  private readonly wn: GPUBuffer // exp(−2πi k / nx), k = 0 … nx/2
  private readonly twY: GPUBuffer // for the ny-point FFT along y

  constructor(
    private readonly device: GPUDevice,
    readonly nx: number,
    readonly ny: number,
    readonly batch = 1,
  ) {
    const maxLen = Math.min(MAX_HALF_LENGTH, device.limits.maxComputeWorkgroupStorageSize / 8)
    const pow2 = (n: number) => n >= 2 && (n & (n - 1)) === 0
    if (!pow2(nx) || nx < 4 || nx / 2 > maxLen) throw new Error(`Real FFT width ${nx} must be a power of two in [4, ${2 * maxLen}]`)
    if (!pow2(ny) || ny > maxLen) throw new Error(`FFT height ${ny} must be a power of two in [2, ${maxLen}]`)

    this.halfWidth = nx / 2 + 1
    this.modes = this.halfWidth * ny
    this.r2c = this.pipeline(rowR2cWgsl)
    this.c2r = this.pipeline(rowC2rWgsl)
    this.rowFft = this.pipeline(rowFftWgsl)
    this.transpose = this.pipeline(transposeWgsl)
    const scratchSize = batch * this.modes * 8
    this.scratchA = this.track(this.device.createBuffer({ size: scratchSize, usage: GPUBufferUsage.STORAGE }))
    this.scratchB = this.track(this.device.createBuffer({ size: scratchSize, usage: GPUBufferUsage.STORAGE }))

    this.twHalf = this.table(nx / 4, (k) => (2 * Math.PI * k) / (nx / 2))
    this.wn = this.table(nx / 2 + 1, (k) => (2 * Math.PI * k) / nx)
    this.twY = this.table(ny / 2, (k) => (2 * Math.PI * k) / ny)
  }

  forward(encoder: GPUCommandEncoder, src: GPUBuffer, dst: GPUBuffer): void {
    this.run(encoder, src, dst, 'forward')
  }

  inverse(encoder: GPUCommandEncoder, src: GPUBuffer, dst: GPUBuffer): void {
    this.run(encoder, src, dst, 'inverse')
  }

  destroy(): void {
    for (const b of this.owned) b.destroy()
  }

  private run(encoder: GPUCommandEncoder, src: GPUBuffer, dst: GPUBuffer, dir: 'forward' | 'inverse') {
    if (src === dst) throw new Error('RealFft2D requires distinct src and dst buffers')
    let byDst = this.cache.get(src)
    if (!byDst) this.cache.set(src, (byDst = new WeakMap()))
    let passes = byDst.get(dst)
    if (!passes) byDst.set(dst, (passes = { forward: this.buildForward(src, dst), inverse: this.buildInverse(src, dst) }))

    const pass = encoder.beginComputePass()
    for (const p of passes[dir]) {
      pass.setPipeline(p.pipeline)
      pass.setBindGroup(0, p.bindGroup)
      pass.dispatchWorkgroups(...p.workgroups)
    }
    pass.end()
  }

  private buildForward(src: GPUBuffer, dst: GPUBuffer): Pass[] {
    const { nx, ny, batch, halfWidth } = this
    return [
      this.pass(this.r2c, [this.rowParams(nx, Math.log2(nx / 2), 0, 1), src, this.scratchA, this.twHalf, this.wn], [ny * batch, 1, 1]),
      // [y][kx] (ny × halfWidth)  →  [kx][y]
      this.pass(
        this.transpose,
        [this.transposeParams(ny, halfWidth), this.scratchA, this.scratchB],
        [Math.ceil(halfWidth / 16), Math.ceil(ny / 16), batch],
      ),
      this.pass(this.rowFft, [this.rowParams(ny, Math.log2(ny), 0, 1), this.scratchB, dst, this.twY], [halfWidth * batch, 1, 1]),
    ]
  }

  private buildInverse(src: GPUBuffer, dst: GPUBuffer): Pass[] {
    const { nx, ny, batch, halfWidth } = this
    return [
      this.pass(this.rowFft, [this.rowParams(ny, Math.log2(ny), 1, 1 / ny), src, this.scratchA, this.twY], [halfWidth * batch, 1, 1]),
      // [kx][y] (halfWidth × ny)  →  [y][kx]
      this.pass(
        this.transpose,
        [this.transposeParams(halfWidth, ny), this.scratchA, this.scratchB],
        [Math.ceil(ny / 16), Math.ceil(halfWidth / 16), batch],
      ),
      this.pass(this.c2r, [this.rowParams(nx, Math.log2(nx / 2), 0, 2 / nx), this.scratchB, dst, this.twHalf, this.wn], [ny * batch, 1, 1]),
    ]
  }

  private pass(pipeline: GPUComputePipeline, buffers: GPUBuffer[], workgroups: [number, number, number]): Pass {
    return {
      pipeline,
      workgroups,
      bindGroup: this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
      }),
    }
  }

  /**
   * Row-kernel params, laid out identically for rowFft ({n, log2n, rows, inverse, scale}) and
   * rowR2c/rowC2r ({n, log2(n/2), rows, unused, scale}).
   */
  private rowParams(n: number, log2: number, inverse: number, scale: number): GPUBuffer {
    return this.uniform((v) => {
      v.setUint32(0, n, true)
      v.setUint32(4, log2, true)
      v.setUint32(12, inverse, true)
      v.setFloat32(16, scale, true)
    })
  }

  private transposeParams(rows: number, cols: number): GPUBuffer {
    return this.uniform((v) => {
      v.setUint32(0, rows, true)
      v.setUint32(4, cols, true)
    })
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
    const buffer = this.track(this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }))
    this.device.queue.writeBuffer(buffer, 0, data)
    return buffer
  }

  /** Table of exp(−i·angle(k)) for k < count, computed in f64. */
  private table(count: number, angle: (k: number) => number): GPUBuffer {
    const data = new Float32Array(2 * Math.max(count, 2))
    for (let k = 0; k < count; k++) {
      data[2 * k] = Math.cos(angle(k))
      data[2 * k + 1] = -Math.sin(angle(k))
    }
    const buffer = this.track(this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }))
    this.device.queue.writeBuffer(buffer, 0, data)
    return buffer
  }
}
