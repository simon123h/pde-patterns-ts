// Runs the in-browser self-test pages in headless Chrome (software WebGPU through SwiftShader
// unless a GPU is available) and exits non-zero if any of them fails.
//
// usage: node scripts/browser-tests.mjs [fft] [render] [solver] [solver:<group>]
// env:   CHROME=<path to chrome> (default google-chrome), TEST_TIMEOUT=<seconds per page> (default 900)
import { spawn } from 'node:child_process'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createServer } from 'vite'

// `solver:<group>` runs one group of the solver page, e.g. solver:thin-film, solver:cgle, solver:coef
const PAGES = { fft: 'fft-test.html', render: 'render-test.html', solver: 'solver-test.html' }
const selected = process.argv.slice(2)
const names = selected.length ? selected : ['fft', 'render', 'solver']
const pageOf = (name) => PAGES[name.split(':')[0]]
for (const name of names) if (!pageOf(name) || (name.includes(':') && !name.startsWith('solver:'))) throw new Error(`unknown test "${name}", expected one of ${Object.keys(PAGES).join(', ')} or solver:<group>`)

const timeoutMs = 1000 * Number(process.env.TEST_TIMEOUT ?? 900)
const debugPort = 9333
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
const devtools = async (path, init) => (await fetch(`http://127.0.0.1:${debugPort}${path}`, init)).json()

/** Reads the text of the page's #out element through the DevTools protocol. */
function readOutput(webSocketUrl) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(webSocketUrl)
    ws.onerror = () => reject(new Error('devtools connection failed'))
    ws.onopen = () => ws.send(JSON.stringify({ id: 1, method: 'Runtime.evaluate', params: { expression: "document.getElementById('out')?.textContent ?? ''" } }))
    ws.onmessage = (m) => { resolve(JSON.parse(m.data).result.result.value); ws.close() }
  })
}

const server = await createServer({ logLevel: 'error', server: { port: 0, host: '127.0.0.1' } })
await server.listen()
const origin = server.resolvedUrls.local[0].replace(/\/$/, '')

const profile = mkdtempSync(join(tmpdir(), 'pde-chrome-'))
const chrome = spawn(process.env.CHROME ?? 'google-chrome', [
  '--headless=new', '--no-sandbox', '--disable-dev-shm-usage', '--no-first-run',
  '--enable-unsafe-webgpu', '--enable-unsafe-swiftshader',
  `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`, 'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] })
// chrome's own error output, shown if it does not start
let chromeErrors = ''
chrome.stderr.on('data', (chunk) => (chromeErrors = (chromeErrors + chunk).slice(-2000)))
let chromeExit = null
chrome.on('exit', (code, signal) => (chromeExit = signal ?? code))

let failed = false
// a killed run (timeout, ctrl-c) must not leave chrome or the dev server behind
const abort = () => { chrome.kill('SIGKILL'); process.exit(130) }
process.on('SIGTERM', abort)
process.on('SIGINT', abort)
try {
  // many jobs start at once on a CI runner, so give chrome up to a minute to open its debugging port
  for (let i = 0; ; i++) {
    try { await devtools('/json/version'); break } catch {
      if (chromeExit !== null) throw new Error(`chrome exited at start-up (${chromeExit}): ${chromeErrors}`)
      if (i > 300) throw new Error(`chrome did not start within 60 s: ${chromeErrors}`)
      await sleep(200)
    }
  }
  for (const name of names) {
    const [page, group] = name.split(':')
    const target = await devtools(`/json/new?${origin}/${PAGES[page]}${group ? `?only=${group}` : ''}`, { method: 'PUT' })
    const prefix = page.toUpperCase() + '-'
    const started = Date.now()
    let title = ''
    while (Date.now() - started < timeoutMs) {
      await sleep(2000)
      title = (await devtools('/json')).find((t) => t.id === target.id)?.title ?? ''
      if (title.startsWith(prefix)) break
    }
    const output = await readOutput(target.webSocketDebuggerUrl)
    const ok = title === prefix + 'OK'
    failed ||= !ok
    console.log(`::group::${name} self-test ${ok ? 'passed' : 'FAILED'} (${((Date.now() - started) / 1000).toFixed(0)} s)`)
    console.log(output)
    console.log('::endgroup::')
    if (!ok) console.error(`${name}: ${title.startsWith(prefix) ? title : 'timed out'}`)
    await fetch(`http://127.0.0.1:${debugPort}/json/close/${target.id}`) // replies with plain text
  }
} finally {
  chrome.kill('SIGKILL')
  await server.close()
  await sleep(500) // let chrome's child processes release the profile directory
  rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 300 })
}
process.exit(failed ? 1 : 0)
