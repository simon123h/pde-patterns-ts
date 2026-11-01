import { Fft2D } from '../gpu/fft'
import { RealFft2D } from '../gpu/realFft'
import type { SpectralTransform } from '../gpu/transform'
import type { SpectralModel } from '../models/types'
import coefficientsWgsl from './shaders/coefficients.wgsl?raw'
import combineWgsl from './shaders/combine.wgsl?raw'
import commitWgsl from './shaders/commit.wgsl?raw'
import controllerWgsl from './shaders/controller.wgsl?raw'
import { StateGuard } from './stateGuard'
import { derivativeCount, derivativeShader } from './derivativeShader'
import paintWgsl from './shaders/paint.wgsl?raw'
import { buildCoefficients, buildSymbols, COEFFICIENT_PLANES, spectralModes, type GridSpec } from './coefficients'
import { nonlinearShader } from './nonlinearShader'

const WORKGROUP = 256
/** Halvings of the step size in a row (without a healthy check in between) before giving up. */
const MAX_CONSECUTIVE_RECOVERIES = 8
const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
/** Most brush stamps applied in one frame. */
export const MAX_STAMPS = 256
const UNIFORM = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST

export interface AdaptiveOptions {
  enabled: boolean
  /** Relative error tolerance per step. */
  tol: number
  /** Largest allowed step size; the smallest is `hmin`. */
  hmax: number
  hmin: number
}

/** Controller state mirrored from the GPU (asynchronously, a frame or two behind). */
export interface StepStats {
  h: number
  t: number
  error: number
  accepted: number
  rejected: number
}

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
  private cpuTime = 0
  private adaptive: AdaptiveOptions = { enabled: false, tol: 1e-3, hmax: 1, hmin: 1e-4 }
  private symbolsDirty = true
  private latestStats: StepStats = { h: 0, t: 0, error: 0, accepted: 0, rejected: 0 }
  private statsMapping = false
  private statsPending = false

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
  private readonly combineAdaptiveBind: GPUBindGroup
  /** Spectral derivatives of the state for the nonlinearity: spectra, then their physical-space fields. */
  private readonly derivatives?: {
    pipeline: GPUComputePipeline
    binds: GPUBindGroup[]
    uniform: GPUBuffer
    hats: GPUBuffer[]
    physical: GPUBuffer[]
  }
  private readonly adapt: {
    symbols: GPUBuffer
    ctl: GPUBuffer
    ctlParams: GPUBuffer
    args: GPUBuffer
    staging: GPUBuffer
    partialCount: number
    coefficients: GPUComputePipeline
    coefficientBind: GPUBindGroup
    controller: GPUComputePipeline
    controllerBind: GPUBindGroup
    commit: GPUComputePipeline
    commitBind: GPUBindGroup
  }
  private readonly guard: StateGuard
  private recoveryCount = 0
  private consecutiveRecoveries = 0
  private gaveUp = false
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
    if (model.usesGradient || model.usesHigherDerivatives) {
      const count = derivativeCount(!!model.usesHigherDerivatives)
      const derivativePipeline = pipeline(derivativeShader(!!model.usesHigherDerivatives))
      const uniform = buffer(32, UNIFORM)
      const hats = Array.from({ length: count }, () => buffer(specBytes, STORAGE))
      const physical = Array.from({ length: count }, () => buffer(physBytes, STORAGE))
      this.derivatives = {
        pipeline: derivativePipeline,
        uniform,
        hats,
        physical,
        binds: [this.uHat, this.stageState].map((src) => bind(derivativePipeline, [uniform, src, ...hats])),
      }
      nonlinearBuffers.push(...physical)
      this.writeDerivativeUniform()
    }
    this.nonlinearBind = bind(this.nonlinearPipeline, nonlinearBuffers)

    this.combinePipeline = pipeline(combineWgsl)
    const partialCount = Math.ceil(this.spectralTotal / WORKGROUP)
    const partials = buffer(partialCount * 8, STORAGE)
    // two uniform sets for the final stage: without and with error estimation
    const stageParams = (stage: number, adapt: number) => {
      const b = buffer(16, UNIFORM)
      device.queue.writeBuffer(b, 0, new Uint32Array([stage, this.spectralTotal, adapt, 0]))
      return b
    }
    const combineBind = (params: GPUBuffer) =>
      bind(this.combinePipeline, [params, this.coef, this.uHat, ...this.nHat, this.stageState, partials])
    this.combineBinds = [0, 1, 2, 3].map((stage) => combineBind(stageParams(stage, 0)))
    this.combineAdaptiveBind = combineBind(stageParams(3, 1))

    const ctl = buffer(32, STORAGE)
    const ctlParams = buffer(32, UNIFORM)
    const args = buffer(16, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST)
    const symbols = buffer(this.spectralTotal * 16, STORAGE)
    const coefficientUniform = buffer(16, UNIFORM)
    device.queue.writeBuffer(coefficientUniform, 0, new Uint32Array([this.spectralTotal, 0, 0, 0]))
    const commitUniform = buffer(16, UNIFORM)
    device.queue.writeBuffer(commitUniform, 0, new Uint32Array([this.spectralTotal, 0, 0, 0]))
    const coefficients = pipeline(coefficientsWgsl)
    const controller = pipeline(controllerWgsl)
    const commit = pipeline(commitWgsl)
    this.adapt = {
      symbols,
      ctl,
      ctlParams,
      args,
      staging: buffer(32, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ),
      partialCount,
      coefficients,
      coefficientBind: bind(coefficients, [coefficientUniform, symbols, ctl, this.coef]),
      controller,
      controllerBind: bind(controller, [ctlParams, partials, ctl, args]),
      commit,
      commitBind: bind(commit, [commitUniform, ctl, this.stageState, this.uHat]),
    }

    this.guard = new StateGuard(device, this.uHat, ctl, this.spectralTotal)

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

  /** Simulated time (in adaptive mode mirrored from the GPU, a frame or two behind). */
  get time(): number {
    return this.adaptive.enabled ? this.latestStats.t : this.cpuTime
  }

  /** Current step size (in adaptive mode the controller's latest choice). */
  get dt(): number {
    return this.adaptive.enabled && this.latestStats.h > 0 ? this.latestStats.h : this.h
  }

  get stats(): StepStats {
    return this.latestStats
  }

  /** Side length of the domain. */
  get domain(): number {
    return this.grid.dx * this.grid.nx
  }

  setDt(h: number): void {
    this.gaveUp = false
    this.consecutiveRecoveries = 0
    if (this.adaptive.enabled) {
      // the controller owns h; a manual change restarts it from the requested value
      this.h = h
      this.initController(h, this.latestStats.t)
      return
    }
    if (h === this.h) return
    this.h = h
    this.coefficientsDirty = true
  }

  /** Turn adaptive time stepping on or off and set its tolerance and step-size range. */
  setAdaptive(options: AdaptiveOptions): void {
    const was = this.adaptive.enabled
    this.adaptive = { ...options }
    const view = new DataView(new ArrayBuffer(32))
    view.setUint32(0, this.adapt.partialCount, true)
    view.setUint32(4, Math.ceil(this.spectralTotal / WORKGROUP), true)
    view.setFloat32(8, options.tol, true)
    view.setFloat32(12, options.hmin, true)
    view.setFloat32(16, options.hmax, true)
    view.setFloat32(20, 0.9, true)
    this.device.queue.writeBuffer(this.adapt.ctlParams, 0, view.buffer)
    if (options.enabled && !was) {
      this.symbolsDirty = true
      this.initController(this.h, this.cpuTime)
    } else if (!options.enabled && was) {
      if (this.latestStats.h > 0) this.h = this.latestStats.h
      this.cpuTime = this.latestStats.t
      this.coefficientsDirty = true
    } else if (options.enabled) {
      this.forceCoefficients()
    }
  }

  setDomain(length: number): void {
    const dx = length / this.grid.nx
    if (dx === this.grid.dx) return
    this.grid.dx = dx
    this.coefficientsDirty = true
    this.symbolsDirty = true
    this.writeDerivativeUniform()
    this.writeParams()
  }

  setParams(params: Record<string, number>): void {
    this.params = { ...this.params, ...params }
    this.coefficientsDirty = true
    this.symbolsDirty = true
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
    this.guard.markSnapshot(enc, { h: this.h, t: 0 })
    this.device.queue.submit([enc.finish()])
    this.cpuTime = 0
    this.consecutiveRecoveries = 0
    this.gaveUp = false
    if (this.adaptive.enabled) this.initController(Math.min(Math.max(this.h, this.adaptive.hmin), this.adaptive.hmax), 0)
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
    const adaptive = this.adaptive.enabled
    if (adaptive) this.refreshSymbols()
    else this.refreshCoefficients()
    const g = this.derivatives
    for (let n = 0; n < count; n++) {
      if (adaptive) {
        // recompute the coefficients if the controller changed h (indirect, zero workgroups otherwise)
        const pass = encoder.beginComputePass()
        pass.setPipeline(this.adapt.coefficients)
        pass.setBindGroup(0, this.adapt.coefficientBind)
        pass.dispatchWorkgroupsIndirect(this.adapt.args, 0)
        pass.end()
      }
      for (let stage = 0; stage < 4; stage++) {
        this.fft.inverse(encoder, stage === 0 ? this.uHat : this.stageState, this.physBuffer)
        if (g) {
          this.dispatch(encoder, g.pipeline, g.binds[stage === 0 ? 0 : 1]!, this.spectralTotal)
          g.hats.forEach((hat, i) => this.fft.inverse(encoder, hat, g.physical[i]!))
        }
        this.dispatch(encoder, this.nonlinearPipeline, this.nonlinearBind, this.cells)
        this.fft.forward(encoder, this.physN, this.nHat[stage]!)
        const combine = adaptive && stage === 3 ? this.combineAdaptiveBind : this.combineBinds[stage]!
        this.dispatch(encoder, this.combinePipeline, combine, this.spectralTotal)
      }
      if (adaptive) {
        const pass = encoder.beginComputePass()
        pass.setPipeline(this.adapt.controller)
        pass.setBindGroup(0, this.adapt.controllerBind)
        pass.dispatchWorkgroups(1)
        pass.end()
        this.dispatch(encoder, this.adapt.commit, this.adapt.commitBind, this.spectralTotal)
      } else {
        encoder.copyBufferToBuffer(this.stageState, 0, this.uHat, 0, this.spectralTotal * 8)
        this.cpuTime += this.h
      }
    }
    this.guard.advance(count)
  }

  /**
   * Encode a health check of the state if one is due (after `step`, before submitting). When the state
   * has blown up, `afterSubmit` rolls it back to the last healthy snapshot and halves the step size.
   */
  encodeHealthCheck(encoder: GPUCommandEncoder): void {
    this.guard.encodeCheck(encoder)
  }

  /** Number of times the state exploded and was rolled back with a smaller step size. */
  get recoveries(): number {
    return this.recoveryCount
  }

  /** The state kept exploding even with much smaller steps: stepping should stop. */
  get unstable(): boolean {
    return this.gaveUp
  }

  /** In adaptive mode: encode a copy of the controller state for `afterSubmit` to read back. */
  encodeStatsCopy(encoder: GPUCommandEncoder): void {
    if (!this.adaptive.enabled || this.statsMapping) return
    encoder.copyBufferToBuffer(this.adapt.ctl, 0, this.adapt.staging, 0, 32)
    this.statsPending = true
  }

  /** Call right after submitting the encoder passed to `encodeStatsCopy`. */
  afterSubmit(): Promise<void> {
    const health = this.guard.afterSubmit(this.cpuTime, this.h, this.adaptive.enabled).then((verdict) => {
      if (verdict === 'exploded') this.recover()
      else if (verdict === 'healthy') this.consecutiveRecoveries = 0
    })
    if (!this.statsPending) return health
    this.statsPending = false
    this.statsMapping = true
    const staging = this.adapt.staging
    const stats = staging
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const f = new Float32Array(staging.getMappedRange().slice(0))
        staging.unmap()
        this.latestStats = { h: f[0]!, t: f[1]!, error: f[3]!, accepted: f[4]!, rejected: f[5]! }
      })
      .catch(() => {})
      .finally(() => (this.statsMapping = false))
    return Promise.all([health, stats]).then(() => {})
  }

  /** Roll the state back to the last healthy snapshot and continue with half the step size. */
  private recover(): void {
    this.recoveryCount++
    this.consecutiveRecoveries++
    const snapshot = this.guard.snapshotState
    const enc = this.device.createCommandEncoder()
    this.guard.restore(enc)
    this.device.queue.submit([enc.finish()])
    const h = (this.adaptive.enabled && this.latestStats.h > 0 ? this.latestStats.h : this.h) / 2
    if (this.consecutiveRecoveries > MAX_CONSECUTIVE_RECOVERIES) {
      this.gaveUp = true
      return
    }
    this.h = h
    if (this.adaptive.enabled) {
      this.initController(h, snapshot.t)
    } else {
      this.cpuTime = snapshot.t
      this.coefficientsDirty = true
    }
  }

  /** Read the ETDRK4 coefficient planes back (complex, interleaved); for tests. */
  async readCoefficients(): Promise<Float32Array> {
    const bytes = COEFFICIENT_PLANES * this.spectralTotal * 8
    const staging = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
    const enc = this.device.createCommandEncoder()
    enc.copyBufferToBuffer(this.coef, 0, staging, 0, bytes)
    this.device.queue.submit([enc.finish()])
    await staging.mapAsync(GPUMapMode.READ)
    const out = new Float32Array(staging.getMappedRange().slice(0))
    staging.unmap()
    staging.destroy()
    return out
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
    this.guard.destroy()
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

  private writeDerivativeUniform(): void {
    if (!this.derivatives) return
    const view = new DataView(new ArrayBuffer(32))
    view.setUint32(0, this.grid.nx, true)
    view.setUint32(4, this.grid.ny, true)
    view.setUint32(8, this.modes, true)
    view.setUint32(12, this.spectralTotal, true)
    view.setFloat32(16, this.grid.dx, true)
    this.device.queue.writeBuffer(this.derivatives.uniform, 0, view.buffer)
  }

  /** Reset the GPU controller state (h, t, accepted, error, counters) and recompute the coefficients. */
  private initController(h: number, t: number): void {
    this.device.queue.writeBuffer(this.adapt.ctl, 0, new Float32Array([h, t, 1, 0, 0, 0, 0, 0]))
    this.latestStats = { h, t, error: 0, accepted: 0, rejected: 0 }
    this.forceCoefficients()
  }

  private forceCoefficients(): void {
    const workgroups = Math.ceil(this.spectralTotal / WORKGROUP)
    this.device.queue.writeBuffer(this.adapt.args, 0, new Uint32Array([workgroups, 1, 1, 0]))
  }

  private refreshSymbols(): void {
    if (!this.symbolsDirty) return
    this.device.queue.writeBuffer(this.adapt.symbols, 0, buildSymbols(this.model, this.params, this.grid))
    this.symbolsDirty = false
    this.forceCoefficients()
  }

  private refreshCoefficients(): void {
    if (!this.coefficientsDirty) return
    this.device.queue.writeBuffer(this.coef, 0, buildCoefficients(this.model, this.params, this.grid, this.h))
    this.coefficientsDirty = false
  }
}
