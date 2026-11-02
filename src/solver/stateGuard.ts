const WORKGROUP = 256
const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST

/** The state counts as exploded when ‖û‖² (about the mean of u², per field) is not below this or not finite. */
export const BLOWUP_LIMIT = 1e10

/** Steps between two health checks. */
const CHECK_INTERVAL = 16

const partialShader = /* wgsl */ `
struct P { total: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> uh: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> partials: array<f32>;
var<workgroup> sh: array<f32, ${WORKGROUP}>;

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  var s = 0.0;
  if (gid.x < p.total) {
    let z = uh[gid.x];
    s = dot(z, z);
  }
  sh[lid.x] = s;
  workgroupBarrier();
  for (var stride = ${WORKGROUP / 2}u; stride > 0u; stride = stride / 2u) {
    if (lid.x < stride) {
      sh[lid.x] = sh[lid.x] + sh[lid.x + stride];
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    partials[wid.x] = sh[0];
  }
}
`

// sums the partial sums; NaN or infinity anywhere in the state propagates to the result
const finalShader = /* wgsl */ `
struct P { count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> partials: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
var<workgroup> sh: array<f32, ${WORKGROUP}>;

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  var s = 0.0;
  for (var i = lid.x; i < p.count; i = i + ${WORKGROUP}u) {
    s = s + partials[i];
  }
  sh[lid.x] = s;
  workgroupBarrier();
  for (var stride = ${WORKGROUP / 2}u; stride > 0u; stride = stride / 2u) {
    if (lid.x < stride) {
      sh[lid.x] = sh[lid.x] + sh[lid.x + stride];
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    result[0] = sh[0];
  }
}
`

/** Outcome of a health check: `none` if no check was due, `stale` if its state is gone already. */
export type Verdict = 'none' | 'healthy' | 'exploded' | 'stale'

/** What the solver needs to know to roll back: the step size and time of a snapshot. */
export interface Snapshot {
  h: number
  t: number
}

/**
 * Watches the spectral state for blow-up and keeps a snapshot of an earlier healthy state.
 *
 * Every few steps a reduction computes ‖û‖² on the GPU and copies the state to a candidate buffer;
 * the 16-byte result (norm, controller h and t) is read back asynchronously. A healthy candidate
 * becomes the snapshot at the next check, so the snapshot is always a state that was verified.
 */
export class StateGuard {
  private readonly candidate: GPUBuffer
  private readonly snapshot: GPUBuffer
  private readonly partials: GPUBuffer
  private readonly result: GPUBuffer
  private readonly staging: GPUBuffer
  private readonly partialPipeline: GPUComputePipeline
  private readonly finalPipeline: GPUComputePipeline
  private readonly partialBind: GPUBindGroup
  private readonly finalBind: GPUBindGroup
  private readonly groups: number
  private readonly owned: GPUBuffer[] = []

  private stepsSince = 0
  private encoded = false // a check was encoded and its result is not read yet
  private reading = false
  private promote = false // the last candidate was healthy and replaces the snapshot at the next check
  private epoch = 0 // invalidates checks that were in flight when the state was reset or rolled back
  private candidateInfo: Snapshot = { h: 0, t: 0 }
  private snapshotInfo: Snapshot = { h: 0, t: 0 }

  constructor(
    device: GPUDevice,
    private readonly state: GPUBuffer,
    private readonly controller: GPUBuffer,
    private readonly total: number,
  ) {
    const buffer = (size: number, usage: number) => {
      const b = device.createBuffer({ size, usage })
      this.owned.push(b)
      return b
    }
    this.groups = Math.ceil(total / WORKGROUP)
    this.candidate = buffer(total * 8, STORAGE)
    this.snapshot = buffer(total * 8, STORAGE)
    this.partials = buffer(this.groups * 4, STORAGE)
    this.result = buffer(4, STORAGE)
    this.staging = buffer(16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ)
    const pipeline = (code: string) =>
      device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } })
    const uniform = (value: number) => {
      const b = buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
      device.queue.writeBuffer(b, 0, new Uint32Array([value, 0, 0, 0]))
      return b
    }
    const bind = (p: GPUComputePipeline, buffers: GPUBuffer[]) =>
      device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: buffers.map((b, binding) => ({ binding, resource: { buffer: b } })) })
    this.partialPipeline = pipeline(partialShader)
    this.finalPipeline = pipeline(finalShader)
    this.partialBind = bind(this.partialPipeline, [uniform(total), state, this.partials])
    this.finalBind = bind(this.finalPipeline, [uniform(this.groups), this.partials, this.result])
  }

  /** Time and step size of the snapshot the state would be rolled back to. */
  get snapshotState(): Snapshot {
    return this.snapshotInfo
  }

  /** The state was replaced (reset): the current state becomes the snapshot. */
  markSnapshot(encoder: GPUCommandEncoder, info: Snapshot): void {
    encoder.copyBufferToBuffer(this.state, 0, this.snapshot, 0, this.total * 8)
    this.snapshotInfo = { ...info }
    this.discardPending()
  }

  /** Forget the check in flight and the candidate (protection switched off or the state replaced). */
  discard(): void {
    this.discardPending()
  }

  /** Counts `steps` solver steps towards the next check. */
  advance(steps: number): void {
    this.stepsSince += steps
  }

  /** Encodes a health check if one is due and none is in flight. Call `afterSubmit` after submitting. */
  encodeCheck(encoder: GPUCommandEncoder): void {
    if (this.stepsSince < CHECK_INTERVAL || this.encoded || this.reading) return
    if (this.promote) {
      encoder.copyBufferToBuffer(this.candidate, 0, this.snapshot, 0, this.total * 8)
      this.snapshotInfo = { ...this.candidateInfo }
      this.promote = false
    }
    let pass = encoder.beginComputePass()
    pass.setPipeline(this.partialPipeline)
    pass.setBindGroup(0, this.partialBind)
    pass.dispatchWorkgroups(this.groups)
    pass.end()
    pass = encoder.beginComputePass()
    pass.setPipeline(this.finalPipeline)
    pass.setBindGroup(0, this.finalBind)
    pass.dispatchWorkgroups(1)
    pass.end()
    encoder.copyBufferToBuffer(this.result, 0, this.staging, 0, 4)
    encoder.copyBufferToBuffer(this.controller, 0, this.staging, 4, 8)
    encoder.copyBufferToBuffer(this.state, 0, this.candidate, 0, this.total * 8)
    this.encoded = true
    this.stepsSince = 0
  }

  /**
   * Reads the result of a check encoded in the submission that was just made. `time` and `h` describe
   * the checked state for a later rollback (fixed step size; in adaptive mode the controller's own h and t
   * are read back instead). A verdict is stale if the state was reset or rolled back in the meantime.
   */
  afterSubmit(time: number, h: number, adaptive: boolean): Promise<Verdict> {
    if (!this.encoded) return Promise.resolve('none')
    this.encoded = false
    this.reading = true
    const epoch = this.epoch
    const candidate = { h, t: time }
    return this.staging
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const f = new Float32Array(this.staging.getMappedRange().slice(0))
        this.staging.unmap()
        this.reading = false
        if (epoch !== this.epoch) return 'stale'
        const exploded = !(f[0]! <= BLOWUP_LIMIT)
        // in adaptive mode the controller's own h and t describe the candidate
        this.candidateInfo = adaptive && f[1]! > 0 ? { h: f[1]!, t: f[2]! } : candidate
        this.promote = !exploded
        return exploded ? 'exploded' : 'healthy'
      })
      .catch(() => {
        this.reading = false
        return 'stale'
      })
  }

  /** Encodes the restore of the snapshot into the state. */
  restore(encoder: GPUCommandEncoder): void {
    encoder.copyBufferToBuffer(this.snapshot, 0, this.state, 0, this.total * 8)
    this.discardPending()
  }

  destroy(): void {
    this.owned.forEach((b) => b.destroy())
  }

  private discardPending(): void {
    this.epoch++
    this.promote = false
    this.encoded = false
    this.stepsSince = 0
  }
}
