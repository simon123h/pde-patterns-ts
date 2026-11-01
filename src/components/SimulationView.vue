<script setup lang="ts">
import { onBeforeUnmount, onMounted, reactive, ref, shallowRef, watch } from 'vue'
import { requestGpu } from '../gpu/device'
import { models } from '../models'
import type { SpectralModel } from '../models/types'
import { COLORMAPS } from '../render/colormaps'
import { FieldRenderer } from '../render/fieldRenderer'
import { SpectralSolver } from '../solver/spectralSolver'

const GRID_SIZES = [64, 128, 256, 512, 1024]

const canvas = ref<HTMLCanvasElement | null>(null)
const error = ref<string | null>(null)
const adapterLabel = ref('')

const model = shallowRef<SpectralModel>(models[0]!)
// `?grid=64` selects the initial grid size (handy on slow software adapters)
const requestedGrid = Number(new URLSearchParams(location.search).get('grid'))
const gridSize = ref(GRID_SIZES.includes(requestedGrid) || requestedGrid === 64 ? requestedGrid : 256)
const dt = ref(model.value.dt.default)
const domain = ref(model.value.domain.default)
const stepsPerFrame = ref(model.value.stepsPerFrame)
const viewIndex = ref(0)
const colormap = ref('viridis')
const running = ref(true)
const params = reactive<Record<string, number>>({})
const stats = reactive({ fps: 0, time: 0 })

let device: GPUDevice | null = null
let solver: SpectralSolver | null = null
let renderer: FieldRenderer | null = null
let raf = 0
let lastFrame = 0
let inFlight = 0 // submitted frames the GPU hasn't finished; limits queue build-up on slow adapters

function loadModelDefaults() {
  const m = model.value
  for (const key of Object.keys(params)) delete params[key]
  for (const p of m.params) params[p.key] = p.default
  dt.value = m.dt.default
  domain.value = m.domain.default
  stepsPerFrame.value = m.stepsPerFrame
  viewIndex.value = 0
}

function selectModel(id: string) {
  model.value = models.find((m) => m.id === id)!
  loadModelDefaults()
  build()
}

function build() {
  if (!device || !canvas.value) return
  solver?.destroy()
  renderer?.destroy()
  const grid = { nx: gridSize.value, ny: gridSize.value, dx: domain.value / gridSize.value }
  solver = new SpectralSolver(device, model.value, grid)
  solver.setParams({ ...params })
  solver.setDt(dt.value)
  solver.setDomain(domain.value)
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
  if (!device || !solver || !renderer || inFlight >= 2) return
  resizeCanvas()
  const enc = device.createCommandEncoder()
  if (running.value) solver.step(enc, stepsPerFrame.value)
  solver.encodeView(enc)
  const view = model.value.views[viewIndex.value] ?? model.value.views[0]!
  renderer.render(enc, view.field, view.kind, colormap.value)
  device.queue.submit([enc.finish()])
  inFlight++
  void device.queue.onSubmittedWorkDone().then(() => inFlight--)

  if (lastFrame) stats.fps = stats.fps * 0.9 + (1000 / (now - lastFrame)) * 0.1
  lastFrame = now
  stats.time = solver.time
}

onMounted(async () => {
  loadModelDefaults()
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
watch(domain, (v) => solver?.setDomain(v))
watch(params, () => solver?.setParams({ ...params }), { deep: true })
// parameters that only shape the initial condition restart the run
watch(
  () => model.value.params.filter((p) => p.resetOnChange).map((p) => params[p.key]),
  () => resetSimulation(),
)
</script>

<template>
  <div class="layout">
    <div class="stage"><canvas ref="canvas" class="view" /></div>
    <aside class="panel">
      <h1>PDE Patterns</h1>
      <p v-if="error" class="error">{{ error }}</p>
      <p v-else class="muted">{{ adapterLabel || 'WebGPU' }} · {{ stats.fps.toFixed(0) }} fps · t = {{ stats.time.toFixed(1) }}</p>

      <label>
        <span>equation</span>
        <select :value="model.id" @change="selectModel(($event.target as HTMLSelectElement).value)">
          <option v-for="m in models" :key="m.id" :value="m.id">{{ m.name }}</option>
        </select>
      </label>
      <p class="muted desc">{{ model.description }}</p>

      <h2>Parameters</h2>
      <label v-for="p in model.params" :key="p.key">
        <span>{{ p.label }} <output>{{ params[p.key]?.toFixed(2) }}</output></span>
        <input v-model.number="params[p.key]" type="range" :min="p.min" :max="p.max" :step="p.step" />
      </label>

      <h2>Domain &amp; numerics</h2>
      <label>
        <span>domain size L <output>{{ domain.toFixed(model.domain.step < 1 ? 1 : 0) }}</output></span>
        <input v-model.number="domain" type="range" :min="model.domain.min" :max="model.domain.max" :step="model.domain.step" />
      </label>
      <label>
        <span>dt <output>{{ dt.toFixed(2) }}</output></span>
        <input v-model.number="dt" type="range" :min="model.dt.min" :max="model.dt.max" :step="model.dt.step" />
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
        <small class="muted">dx = {{ (domain / gridSize).toPrecision(3) }}</small>
      </label>
      <label v-if="model.views.length > 1">
        <span>view</span>
        <select v-model.number="viewIndex">
          <option v-for="(v, i) in model.views" :key="i" :value="i">{{ v.label }}</option>
        </select>
      </label>
      <label>
        <span>colormap</span>
        <select v-model="colormap">
          <option v-for="c in COLORMAPS" :key="c.id" :value="c.id">{{ c.label }}</option>
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
.stage { flex: 1; min-width: 0; display: grid; place-items: center; background: #000; }
.view { width: min(100%, 100vh); aspect-ratio: 1; }
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
.desc { font-size: 12px; }
.error { color: #ff7b7b; margin: 0; }
</style>
