import fieldWgsl from './shaders/field.wgsl?raw'
import type { ViewKind } from '../models/types'
import rangeWgsl from './shaders/range.wgsl?raw'

export type Colormap = 'viridis' | 'gray'
const COLORMAP_ID: Record<Colormap, number> = { viridis: 0, gray: 1 }
const KIND_ID: Record<ViewKind, number> = { re: 0, im: 1, abs: 2, arg: 3 }

/** Draws one field of a GPU buffer onto a canvas, auto-scaling the color range on the GPU. */
export class FieldRenderer {
  private readonly context: GPUCanvasContext
  private readonly format: GPUTextureFormat
  private readonly rangePipeline: GPUComputePipeline
  private readonly renderPipeline: GPURenderPipeline
  private readonly rangeParams: GPUBuffer
  private readonly renderParams: GPUBuffer
  private readonly rangeBuffer: GPUBuffer
  private readonly rangeBind: GPUBindGroup
  private readonly renderBind: GPUBindGroup
  private first = true

  constructor(
    private readonly device: GPUDevice,
    private readonly canvas: HTMLCanvasElement,
    field: GPUBuffer,
    private readonly nx: number,
    private readonly ny: number,
  ) {
    const context = canvas.getContext('webgpu')
    if (!context) throw new Error('Could not create a WebGPU canvas context')
    this.context = context
    this.format = navigator.gpu.getPreferredCanvasFormat()
    context.configure({ device, format: this.format, alphaMode: 'opaque' })

    this.rangeParams = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    this.renderParams = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    this.rangeBuffer = device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE })

    this.rangePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: rangeWgsl }), entryPoint: 'main' },
    })
    const fieldModule = device.createShaderModule({ code: fieldWgsl })
    this.renderPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: fieldModule, entryPoint: 'vs' },
      fragment: { module: fieldModule, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    })

    const bind = (pipeline: GPUComputePipeline | GPURenderPipeline, buffers: GPUBuffer[]) =>
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
      })
    this.rangeBind = bind(this.rangePipeline, [this.rangeParams, field, this.rangeBuffer])
    this.renderBind = bind(this.renderPipeline, [this.renderParams, field, this.rangeBuffer])
  }

  /** Forget the smoothed color range (e.g. after a reset) so it re-fits immediately. */
  resetRange(): void {
    this.first = true
  }

  /** Encode range detection and drawing of one component of field `fieldIndex` into the canvas. */
  render(encoder: GPUCommandEncoder, fieldIndex: number, kind: ViewKind, colormap: Colormap): void {
    const cells = this.nx * this.ny
    const offset = fieldIndex * cells
    const stride = Math.max(1, Math.floor(cells / 65536)) | 1 // odd stride avoids lattice aliasing

    const rp = new DataView(new ArrayBuffer(32))
    rp.setUint32(0, cells, true)
    rp.setUint32(4, offset, true)
    rp.setUint32(8, stride, true)
    rp.setUint32(12, this.first ? 1 : 0, true)
    rp.setFloat32(16, 0.15, true)
    rp.setUint32(20, KIND_ID[kind], true)
    this.device.queue.writeBuffer(this.rangeParams, 0, rp.buffer)
    this.first = false

    const up = new DataView(new ArrayBuffer(32))
    up.setUint32(0, this.nx, true)
    up.setUint32(4, this.ny, true)
    up.setUint32(8, offset, true)
    up.setUint32(12, COLORMAP_ID[colormap], true)
    up.setFloat32(16, this.canvas.width, true)
    up.setFloat32(20, this.canvas.height, true)
    up.setUint32(24, KIND_ID[kind], true)
    this.device.queue.writeBuffer(this.renderParams, 0, up.buffer)

    const compute = encoder.beginComputePass()
    compute.setPipeline(this.rangePipeline)
    compute.setBindGroup(0, this.rangeBind)
    compute.dispatchWorkgroups(1)
    compute.end()

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
    })
    pass.setPipeline(this.renderPipeline)
    pass.setBindGroup(0, this.renderBind)
    pass.draw(3)
    pass.end()
  }

  destroy(): void {
    this.context.unconfigure()
    for (const b of [this.rangeParams, this.renderParams, this.rangeBuffer]) b.destroy()
  }
}
