import { Fft2D } from '../gpu/fft'
import type { SpectralModel } from '../models/types'
import combineWgsl from './shaders/combine.wgsl?raw'
import { buildCoefficients, COEFFICIENT_PLANES, type GridSpec } from './coefficients'
import { nonlinearShader } from './nonlinearShader'

const WORKGROUP = 256
const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST

/**
 * Pseudospectral solver for ∂t u = L û + N(u) on a periodic nx × ny grid, advanced with the
 * Krogstad ETDRK4 scheme entirely on the GPU. Fields are stored back to back (field f at
 * element offset f·nx·ny); spectral buffers use the transposed layout of `Fft2D`.
 */
export class SpectralSolver {
  readonly cells: number
  readonly total: number
  /** Simulated time. */
  time = 0

  private params: Record<string, number>
  private h = 0.1
  private coefficientsDirty = true

  private readonly fft: Fft2D
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
  private readonly owned: GPUBuffer[] = []

  constructor(
    private readonly device: GPUDevice,
    readonly model: SpectralModel,
    readonly grid: GridSpec,
  ) {
    this.cells = grid.nx * grid.ny
    this.total = this.cells * model.fields
    const bytes = this.total * 8
    if (bytes > device.limits.maxStorageBufferBindingSize) {
      throw new Error(`Grid ${grid.nx}×${grid.ny} with ${model.fields} field(s) exceeds the device buffer limit`)
    }
    if (Math.ceil(this.total / WORKGROUP) > device.limits.maxComputeWorkgroupsPerDimension) {
      throw new Error('Grid too large for a single-dimension dispatch')
    }

    this.params = Object.fromEntries(model.params.map((p) => [p.key, p.default]))
    this.fft = new Fft2D(device, grid.nx, grid.ny, model.fields)

    const buffer = (size: number, usage: number) => {
      const b = device.createBuffer({ size, usage })
      this.owned.push(b)
      return b
    }
    this.physBuffer = buffer(bytes, STORAGE)
    this.physN = buffer(bytes, STORAGE)
    this.uHat = buffer(bytes, STORAGE)
    this.stageState = buffer(bytes, STORAGE)
    this.nHat = [0, 1, 2, 3].map(() => buffer(bytes, STORAGE))
    this.coef = buffer(COEFFICIENT_PLANES * this.total * 4, STORAGE)
    this.paramBuffer = buffer(Math.max(16, Math.ceil(model.params.length / 4) * 16), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)

    this.nonlinearPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: nonlinearShader(model, this.cells) }),
        entryPoint: 'main',
      },
    })
    this.nonlinearBind = device.createBindGroup({
      layout: this.nonlinearPipeline.getBindGroupLayout(0),
      entries: [this.paramBuffer, this.physBuffer, this.physN].map((b, binding) => ({ binding, resource: { buffer: b } })),
    })

    this.combinePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: combineWgsl }), entryPoint: 'main' },
    })
    this.combineBinds = [0, 1, 2, 3].map((stage) => {
      const stageParams = buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
      device.queue.writeBuffer(stageParams, 0, new Uint32Array([stage, this.total, 0, 0]))
      const buffers = [stageParams, this.coef, this.uHat, ...this.nHat, this.stageState]
      return device.createBindGroup({
        layout: this.combinePipeline.getBindGroupLayout(0),
        entries: buffers.map((b, binding) => ({ binding, resource: { buffer: b } })),
      })
    })

    this.writeParams()
  }

  /** Buffer holding the physical-space fields after `encodeView` (field f at offset f·cells). */
  get physical(): GPUBuffer {
    return this.physBuffer
  }

  get dt(): number {
    return this.h
  }

  setDt(h: number): void {
    if (h === this.h) return
    this.h = h
    this.coefficientsDirty = true
  }

  setParams(params: Record<string, number>): void {
    this.params = { ...this.params, ...params }
    this.coefficientsDirty = true
    this.writeParams()
  }

  /** Re-initialize with white noise (or the given physical-space data) and reset time. */
  reset(initial?: Float32Array): void {
    const data = new Float32Array(2 * this.total) // interleaved re/im, im = 0
    for (let i = 0; i < this.total; i++) {
      data[2 * i] = initial ? initial[i]! : this.model.noiseAmplitude * gaussian()
    }
    this.device.queue.writeBuffer(this.physBuffer, 0, data)
    const enc = this.device.createCommandEncoder()
    this.fft.forward(enc, this.physBuffer, this.uHat)
    this.device.queue.submit([enc.finish()])
    this.time = 0
  }

  /** Encode `count` ETDRK4 steps. Must be submitted before the next call that changes dt or params. */
  step(encoder: GPUCommandEncoder, count = 1): void {
    this.refreshCoefficients()
    for (let n = 0; n < count; n++) {
      for (let stage = 0; stage < 4; stage++) {
        this.fft.inverse(encoder, stage === 0 ? this.uHat : this.stageState, this.physBuffer)
        this.dispatch(encoder, this.nonlinearPipeline, this.nonlinearBind)
        this.fft.forward(encoder, this.physN, this.nHat[stage]!)
        this.dispatch(encoder, this.combinePipeline, this.combineBinds[stage]!)
      }
      encoder.copyBufferToBuffer(this.stageState, 0, this.uHat, 0, this.total * 8)
      this.time += this.h
    }
  }

  /** Transform the current state to physical space into `physical`. */
  encodeView(encoder: GPUCommandEncoder): void {
    this.fft.inverse(encoder, this.uHat, this.physBuffer)
  }

  destroy(): void {
    this.fft.destroy()
    for (const b of this.owned) b.destroy()
  }

  private dispatch(encoder: GPUCommandEncoder, pipeline: GPUComputePipeline, bind: GPUBindGroup): void {
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bind)
    pass.dispatchWorkgroups(Math.ceil(this.total / WORKGROUP))
    pass.end()
  }

  private writeParams(): void {
    const values = new Float32Array(Math.max(4, Math.ceil(this.model.params.length / 4) * 4))
    this.model.params.forEach((p, i) => (values[i] = this.params[p.key]!))
    this.device.queue.writeBuffer(this.paramBuffer, 0, values)
  }

  private refreshCoefficients(): void {
    if (!this.coefficientsDirty) return
    this.device.queue.writeBuffer(this.coef, 0, buildCoefficients(this.model, this.params, this.grid, this.h))
    this.coefficientsDirty = false
  }
}

/** Standard normal sample (Box–Muller). */
function gaussian(): number {
  const u = 1 - Math.random()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random())
}
