<script setup lang="ts">
import { onBeforeUnmount, onMounted, reactive, ref, shallowRef, watch } from 'vue'
import { requestGpu } from '../gpu/device'
import { models } from '../models'
import { FieldRenderer, type Colormap } from '../render/fieldRenderer'
import { SpectralSolver } from '../solver/spectralSolver'

const GRID_SIZES = [128, 256, 512, 1024]
const DX = 0.5

const canvas = ref<HTMLCanvasElement | null>(null)
const error = ref<string | null>(null)
const adapterLabel = ref('')

const model = shallowRef(models[0]!)
const gridSize = ref(256)
const dt = ref(0.2)
const stepsPerFrame = ref(2)
const colormap = ref<Colormap>('viridis')
const running = ref(true)
const params = reactive<Record<string, number>>({})
const stats = reactive({ fps: 0, time: 0 })

let device: GPUDevice | null = null
let solver: SpectralSolver | null = null
let renderer: FieldRenderer | null = null
let raf = 0
let lastFrame = 0

function resetParams() {
  for (const p of model.value.params) params[p.key] = p.default
}

function build() {
  if (!device || !canvas.value) return
  solver?.destroy()
  renderer?.destroy()
  const grid = { nx: gridSize.value, ny: gridSize.value, dx: DX }
  solver = new SpectralSolver(device, model.value, grid)
  solver.setParams({ ...params })
  solver.setDt(dt.value)
  solver.reset()
  renderer = new FieldRenderer(device, canvas.value, solver.physical, grid.nx, grid.ny)
}

function resetSimulation() {
  solver?.reset()
  renderer?.resetRange()
}

function resizeCanvas() {
  const el = canvas.value
  if (!el) return
  const dpr = window.devicePixelRatio || 1
  const size = Math.max(64, Math.floor(el.clientWidth * dpr))
  if (el.width !== size || el.height !== size) {
    el.width = size
    el.height = size
  }
}

function frame(now: number) {
  raf = requestAnimationFrame(frame)
  if (!device || !solver || !renderer) return
  resizeCanvas()
  const enc = device.createCommandEncoder()
  if (running.value) solver.step(enc, stepsPerFrame.value)
  solver.encodeView(enc)
  renderer.render(enc, 0, colormap.value)
  device.queue.submit([enc.finish()])

  if (lastFrame) stats.fps = stats.fps * 0.9 + (1000 / (now - lastFrame)) * 0.1
  lastFrame = now
  stats.time = solver.time
}

onMounted(async () => {
  resetParams()
  try {
    const gpu = await requestGpu()
    device = gpu.device
    const info = gpu.adapter.info
    adapterLabel.value = `${info.vendor} ${info.architecture}`.trim()
    device.lost.then((info) => (error.value = `GPU device lost: ${info.message}`))
    build()
    raf = requestAnimationFrame(frame)
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  }
})

onBeforeUnmount(() => {
  cancelAnimationFrame(raf)
  solver?.destroy()
  renderer?.destroy()
})

watch(gridSize, build)
watch(dt, (v) => solver?.setDt(v))
watch(params, () => solver?.setParams({ ...params }), { deep: true })
</script>

<template>
  <div class="layout">
    <canvas ref="canvas" class="view" />
    <aside class="panel">
      <h1>PDE Patterns</h1>
      <p v-if="error" class="error">{{ error }}</p>
      <p v-else class="muted">{{ adapterLabel || 'WebGPU' }} · {{ stats.fps.toFixed(0) }} fps · t = {{ stats.time.toFixed(1) }}</p>

      <h2>{{ model.name }}</h2>
      <label v-for="p in model.params" :key="p.key">
        <span>{{ p.label }} <output>{{ params[p.key]?.toFixed(2) }}</output></span>
        <input v-model.number="params[p.key]" type="range" :min="p.min" :max="p.max" :step="p.step" />
      </label>

      <h2>Numerics</h2>
      <label>
        <span>dt <output>{{ dt.toFixed(2) }}</output></span>
        <input v-model.number="dt" type="range" min="0.01" max="2" step="0.01" />
      </label>
      <label>
        <span>steps / frame <output>{{ stepsPerFrame }}</output></span>
        <input v-model.number="stepsPerFrame" type="range" min="1" max="20" step="1" />
      </label>
      <label>
        <span>grid</span>
        <select v-model.number="gridSize">
          <option v-for="n in GRID_SIZES" :key="n" :value="n">{{ n }} × {{ n }}</option>
        </select>
      </label>
      <label>
        <span>colormap</span>
        <select v-model="colormap">
          <option value="viridis">viridis</option>
          <option value="gray">gray</option>
        </select>
      </label>

      <div class="buttons">
        <button @click="running = !running">{{ running ? 'Pause' : 'Run' }}</button>
        <button @click="resetSimulation">Reset</button>
      </div>
    </aside>
  </div>
</template>

<style scoped>
.layout { display: flex; height: 100vh; }
.view { flex: 1; min-width: 0; height: 100%; object-fit: contain; background: #000; aspect-ratio: 1; max-width: 100vh; }
.panel { width: 280px; padding: 16px; background: var(--panel); overflow-y: auto; display: flex; flex-direction: column; gap: 10px; }
h1 { font-size: 18px; margin: 0; }
h2 { font-size: 13px; margin: 12px 0 0; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
label { display: flex; flex-direction: column; gap: 4px; }
label span { display: flex; justify-content: space-between; }
output { color: var(--muted); font-variant-numeric: tabular-nums; }
select, button { background: #232734; color: var(--fg); border: 1px solid #333a4d; border-radius: 6px; padding: 6px 10px; }
button { cursor: pointer; flex: 1; }
button:hover { border-color: var(--accent); }
.buttons { display: flex; gap: 8px; margin-top: 8px; }
.muted { color: var(--muted); margin: 0; }
.error { color: #ff7b7b; margin: 0; }
</style>
