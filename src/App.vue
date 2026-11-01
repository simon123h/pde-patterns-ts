<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { requestGpu } from './gpu/device'
import { models } from './models'

const status = ref('Checking WebGPU…')
const error = ref<string | null>(null)

onMounted(async () => {
  try {
    const { adapter } = await requestGpu()
    const info = adapter.info
    // Firefox hides adapter details (fingerprinting protection), so these may be empty.
    const label = `${info.vendor} ${info.architecture}`.trim()
    status.value = `WebGPU ready${label ? `: ${label}` : ' (adapter details hidden by the browser)'}`
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  }
})
</script>

<template>
  <main class="shell">
    <h1>PDE Patterns</h1>
    <p v-if="error" class="error">{{ error }}</p>
    <p v-else class="muted">{{ status }}</p>
    <p class="muted">Models: {{ models.map((m) => m.name).join(', ') }}</p>
  </main>
</template>

<style scoped>
.shell { padding: 24px; }
.muted { color: var(--muted); }
.error { color: #ff7b7b; }
</style>
