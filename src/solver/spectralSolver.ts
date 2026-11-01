import { Fft2D } from '../gpu/fft'
import { RealFft2D } from '../gpu/realFft'
import type { SpectralTransform } from '../gpu/transform'
import type { SpectralModel } from '../models/types'
import combineWgsl from './shaders/combine.wgsl?raw'
import gradientWgsl from './shaders/gradient.wgsl?raw'
import paintWgsl from './shaders/paint.wgsl?raw'
import { buildCoefficients, COEFFICIENT_PLANES, spectralModes, type GridSpec } from './coefficients'
import { nonlinearShader } from './nonlinearShader'

const WORKGROUP = 256
const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
/** Most brush stamps applied in one frame. */
export const MAX_STAMPS = 256
const UNIFORM = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST

/**
 * Pseudospectral solver for ∂t û = L û + M·FFT[N(u)] on a periodic square nx × ny grid,
 * advanced with the Krogstad ETDRK4 scheme entirely on the GPU. Fields are stored back to back
 * (field f at element offset f·nx·ny). Real models keep physical fields as plain f32 and use
 * real-to-complex FFTs (half spectrum kx ≤ nx/2); complex models (CGLE) use vec2<f32> fields and
 * full complex FFTs. Spectral buffers always use the transposed layout (index kx·ny + ky).
 */
export class SpectralSolver {
  /** Grid cells per field. */
  readonly cells: number
  /** Physical values: fields · cells. */
  readonly total: number
  /** Whether the fields are stored as real f32 (otherwise complex vec2<f32>). */
  readonly real: boolean
  /** Spectral modes per field, and over all fields. */
  readonly modes: number
  readonly spectralTotal: number
  /** Simulated time. */
  time = 0

  private readonly grid: GridSpec
  private params: Record<string, number>
  private h: number
  private coefficientsDirty = true

  private readonly fft: SpectralTransform
  private readonly physBuffer: GPUBuffer
  private readonly physN: GPUBuffer
  private readonly uHat: GPUBuffer
  private readonly stageState: GPUBuffer
  private readonly nHat: GPUBuffer[]
  private readonly coef: GPUBuffer
  private readonly paramBuffer: GPUBuffer
  private readonly nonlinearPipeline: GPUComputePipeline
  private readonly nonlinearBind: GPUBindGroup
  private readonly combinePipeline: GPUComputePipeline
  private readonly combineBinds: GPUBindGroup[]
  private readonly gradient?: {
    pipeline: GPUComputePipeline
    binds: GPUBindGroup[]
    uniform: GPUBuffer
    xHat: GPUBuffer
    yHat: GPUBuffer
    xPhys: GPUBuffer
    yPhys: GPUBuffer
  }
  private readonly paintPipeline: GPUComputePipeline
  private readonly paintBind: GPUBindGroup
  private readonly paintUniform: GPUBuffer
  private readonly stampBuffer: GPUBuffer
  private readonly owned: GPUBuffer[] = []

  constructor(
    private readonly device: GPUDevice,
    readonly model: SpectralModel,
    gridSpec: GridSpec,
  ) {
    this.grid = { ...gridSpec }
    this.real = !model.complex
    this.cells = gridSpec.nx * gridSpec.ny
    this.total = this.cells * model.fields
    this.modes = spectralModes(model, gridSpec)
    this.spectralTotal = this.modes * model.fields
    const physBytes = this.total * (this.real ? 4 : 8)
    const specBytes = this.spectralTotal * 8
    const coefBytes = COEFFICIENT_PLANES * specBytes
    if (coefBytes > device.limits.maxStorageBufferBindingSize) {
      throw new Error(`Grid ${gridSpec.nx}×${gridSpec.ny} with ${model.fields} field(s) exceeds the device buffer limit`)
    }
    if (Math.ceil(this.spectralTotal / WORKGROUP) > device.limits.maxComputeWorkgroupsPerDimension) {
      throw new Error('Grid too large for a single-dimension dispatch')
    }

    this.params = Object.fromEntries(model.params.map((p) => [p.key, p.default]))
    this.h = model.dt.default
    this.fft = this.real
      ? new RealFft2D(device, gridSpec.nx, gridSpec.ny, model.fields)
      : new Fft2D(device, gridSpec.nx, gridSpec.ny, model.fields)

    const buffer = (size: number, usage: number) => {
      const b = device.createBuffer({ size, usage })
      this.owned.push(b)
      return b
    }
    const pipeline = (code: string) =>
      device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
      })
    const bind = (p: GPUComputePipeline, buffers: GPUBuffer[]) =>
      device.createBindGroup({
        layout: p.getBindGroupLayout(0),
        entries: buffers.map((b, binding) => ({ binding, resource: { buffer: b } })),
      })

    this.physBuffer = buffer(physBytes, STORAGE)
    this.physN = buffer(physBytes, STORAGE)
    this.uHat = buffer(specBytes, STORAGE)
    this.stageState = buffer(specBytes, STORAGE)
    this.nHat = [0, 1, 2, 3].map(() => buffer(specBytes, STORAGE))
    this.coef = buffer(coefBytes, STORAGE)
    this.paramBuffer = buffer(Math.ceil((model.params.length + 1) / 4) * 16, UNIFORM)

    this.nonlinearPipeline = pipeline(nonlinearShader(model, this.cells, gridSpec.nx))
    const nonlinearBuffers = [this.paramBuffer, this.physBuffer, this.physN]
    if (model.usesGradient) {
      const gradientPipeline = pipeline(gradientWgsl)
      const uniform = buffer(32, UNIFORM)
      const xHat = buffer(specBytes, STORAGE)
      const yHat = buffer(specBytes, STORAGE)
      const xPhys = buffer(physBytes, STORAGE)
      const yPhys = buffer(physBytes, STORAGE)
      this.gradient = {
        pipeline: gradientPipeline,
        uniform,
        xHat,
        yHat,
        xPhys,
        yPhys,
        binds: [this.uHat, this.stageState].map((src) => bind(gradientPipeline, [uniform, src, xHat, yHat])),
      }
      nonlinearBuffers.push(xPhys, yPhys)
      this.writeGradientUniform()
    }
    this.nonlinearBind = bind(this.nonlinearPipeline, nonlinearBuffers)

    this.combinePipeline = pipeline(combineWgsl)
    this.combineBinds = [0, 1, 2, 3].map((stage) => {
      const stageParams = buffer(16, UNIFORM)
      device.queue.writeBuffer(stageParams, 0, new Uint32Array([stage, this.spectralTotal, 0, 0]))
      return bind(this.combinePipeline, [stageParams, this.coef, this.uHat, ...this.nHat, this.stageState])
    })

    this.paintPipeline = pipeline(paintWgsl)
    this.paintUniform = buffer(48, UNIFORM)
    this.stampBuffer = buffer(MAX_STAMPS * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST)
    this.paintBind = bind(this.paintPipeline, [this.paintUniform, this.stampBuffer, this.uHat])

    this.writeParams()
  }

  /** Buffer holding the physical-space fields after `encodeView` (complex, field f at offset f·cells). */
  get physical(): GPUBuffer {
    return this.physBuffer
  }

  get dt(): number {
    return this.h
  }

  /** Side length of the domain. */
  get domain(): number {
    return this.grid.dx * this.grid.nx
  }

  setDt(h: number): void {
    if (h === this.h) return
    this.h = h
    this.coefficientsDirty = true
  }

  setDomain(length: number): void {
    const dx = length / this.grid.nx
    if (dx === this.grid.dx) return
    this.grid.dx = dx
    this.coefficientsDirty = true
    this.writeGradientUniform()
    this.writeParams()
  }

  setParams(params: Record<string, number>): void {
    this.params = { ...this.params, ...params }
    this.coefficientsDirty = true
    this.writeParams()
  }

  /**
   * Re-initialize and reset time. With no argument the model's initial condition is used; `initial`
   * may be real-valued (length fields·cells) or interleaved re/im (length 2·fields·cells).
   */
  reset(initial?: Float32Array): void {
    // normalize to interleaved (re, im)
    let data: Float32Array
    if (!initial) {
      data = this.model.initial({ nx: this.grid.nx, ny: this.grid.ny, lx: this.domain, params: this.params })
    } else if (initial.length === this.total) {
      data = new Float32Array(2 * this.total)
      for (let i = 0; i < this.total; i++) data[2 * i] = initial[i]!
    } else {
      data = initial
    }
    if (data.length !== 2 * this.total) throw new Error('Initial state has the wrong size')

    let upload: Float32Array<ArrayBuffer>
    if (this.real) {
      upload = new Float32Array(this.total)
      for (let i = 0; i < this.total; i++) upload[i] = data[2 * i]!
    } else {
      upload = new Float32Array(data)
    }
    this.device.queue.writeBuffer(this.physBuffer, 0, upload)
    const enc = this.device.createCommandEncoder()
    this.fft.forward(enc, this.physBuffer, this.uHat)
    this.device.queue.submit([enc.finish()])
    this.time = 0
  }

  /** Read the current physical-space fields back as interleaved (re, im), fields back to back. */
  async readPhysical(): Promise<Float32Array> {
    const bytes = this.total * (this.real ? 4 : 8)
    const staging = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
    const enc = this.device.createCommandEncoder()
    this.encodeView(enc)
    enc.copyBufferToBuffer(this.physBuffer, 0, staging, 0, bytes)
    this.device.queue.submit([enc.finish()])
    await staging.mapAsync(GPUMapMode.READ)
    const raw = new Float32Array(staging.getMappedRange().slice(0))
    staging.unmap()
    staging.destroy()
    if (!this.real) return raw
    const out = new Float32Array(2 * this.total)
    for (let i = 0; i < this.total; i++) out[2 * i] = raw[i]!
    return out
  }

  /** Encode `count` ETDRK4 steps. Must be submitted before the next call that changes dt or params. */
  step(encoder: GPUCommandEncoder, count = 1): void {
    this.refreshCoefficients()
    const g = this.gradient
    for (let n = 0; n < count; n++) {
      for (let stage = 0; stage < 4; stage++) {
        this.fft.inverse(encoder, stage === 0 ? this.uHat : this.stageState, this.physBuffer)
        if (g) {
          this.dispatch(encoder, g.pipeline, g.binds[stage === 0 ? 0 : 1]!, this.spectralTotal)
          this.fft.inverse(encoder, g.xHat, g.xPhys)
          this.fft.inverse(encoder, g.yHat, g.yPhys)
        }
        this.dispatch(encoder, this.nonlinearPipeline, this.nonlinearBind, this.cells)
        this.fft.forward(encoder, this.physN, this.nHat[stage]!)
        this.dispatch(encoder, this.combinePipeline, this.combineBinds[stage]!, this.spectralTotal)
      }
      encoder.copyBufferToBuffer(this.stageState, 0, this.uHat, 0, this.spectralTotal * 8)
      this.time += this.h
    }
  }

  /**
   * Encode a brush: Gaussian bumps of radius `sigma` (domain units) at the given stamps
   * (x, y in domain units, signed `scale` per stamp), scaled per field by the model's paint amounts.
   * At most MAX_STAMPS stamps; call at most once per submission.
   */
  paint(encoder: GPUCommandEncoder, stamps: Float32Array<ArrayBuffer>, sigma: number): void {
    const count = Math.min(MAX_STAMPS, stamps.length / 4)
    if (count === 0) return
    const amounts = this.model.paint?.amounts ?? [1]
    const view = new DataView(new ArrayBuffer(48))
    view.setUint32(0, count, true)
    view.setUint32(4, this.grid.nx, true)
    view.setUint32(8, this.grid.ny, true)
    view.setUint32(12, this.modes, true)
    view.setUint32(16, this.spectralTotal, true)
    view.setFloat32(20, sigma, true)
    view.setFloat32(24, this.grid.dx, true)
    for (let f = 0; f < 4; f++) view.setFloat32(32 + 4 * f, amounts[f] ?? 0, true)
    this.device.queue.writeBuffer(this.paintUniform, 0, view.buffer)
    this.device.queue.writeBuffer(this.stampBuffer, 0, stamps, 0, count * 4)
    this.dispatch(encoder, this.paintPipeline, this.paintBind, this.spectralTotal)
  }

  /** Transform the current state to physical space into `physical`. */
  encodeView(encoder: GPUCommandEncoder): void {
    this.fft.inverse(encoder, this.uHat, this.physBuffer)
  }

  destroy(): void {
    this.fft.destroy()
    for (const b of this.owned) b.destroy()
  }

  private dispatch(encoder: GPUCommandEncoder, pipeline: GPUComputePipeline, bind: GPUBindGroup, threads: number): void {
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bind)
    pass.dispatchWorkgroups(Math.ceil(threads / WORKGROUP))
    pass.end()
  }

  private writeParams(): void {
    // model parameters followed by the grid spacing (see nonlinearShader)
    const values = new Float32Array(Math.ceil((this.model.params.length + 1) / 4) * 4)
    this.model.params.forEach((p, i) => (values[i] = this.params[p.key]!))
    values[this.model.params.length] = this.grid.dx
    this.device.queue.writeBuffer(this.paramBuffer, 0, values)
  }

  private writeGradientUniform(): void {
    if (!this.gradient) return
    const view = new DataView(new ArrayBuffer(32))
    view.setUint32(0, this.grid.nx, true)
    view.setUint32(4, this.grid.ny, true)
    view.setUint32(8, this.modes, true)
    view.setUint32(12, this.spectralTotal, true)
    view.setFloat32(16, this.grid.dx, true)
    this.device.queue.writeBuffer(this.gradient.uniform, 0, view.buffer)
  }

  private refreshCoefficients(): void {
    if (!this.coefficientsDirty) return
    this.device.queue.writeBuffer(this.coef, 0, buildCoefficients(this.model, this.params, this.grid, this.h))
    this.coefficientsDirty = false
  }
}
