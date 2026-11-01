/** CPU reference FFTs (power-of-two sizes, f64). Used for tests, not for production stepping. */

/** In-place radix-2 FFT of interleaved-free re/im arrays. `inverse` is unnormalized. */
export function fft1d(re: Float64Array, im: Float64Array, inverse = false): void {
  const n = re.length
  if ((n & (n - 1)) !== 0) throw new Error('FFT length must be a power of two')
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      ;[re[i], re[j]] = [re[j]!, re[i]!]
      ;[im[i], im[j]] = [im[j]!, im[i]!]
    }
  }
  const sign = inverse ? 1 : -1
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (sign * 2 * Math.PI) / len
    const wr = Math.cos(ang)
    const wi = Math.sin(ang)
    for (let i = 0; i < n; i += len) {
      let cr = 1
      let ci = 0
      for (let k = 0; k < len / 2; k++) {
        const a = i + k
        const b = a + len / 2
        const tr = re[b]! * cr - im[b]! * ci
        const ti = re[b]! * ci + im[b]! * cr
        re[b] = re[a]! - tr
        im[b] = im[a]! - ti
        re[a] = re[a]! + tr
        im[a] = im[a]! + ti
        const nr = cr * wr - ci * wi
        ci = cr * wi + ci * wr
        cr = nr
      }
    }
  }
}

/** In-place 2D FFT of an nx × ny row-major field (index = y * nx + x). Inverse is unnormalized. */
export function fft2d(re: Float64Array, im: Float64Array, nx: number, ny: number, inverse = false): void {
  const rowRe = new Float64Array(nx)
  const rowIm = new Float64Array(nx)
  for (let y = 0; y < ny; y++) {
    rowRe.set(re.subarray(y * nx, (y + 1) * nx))
    rowIm.set(im.subarray(y * nx, (y + 1) * nx))
    fft1d(rowRe, rowIm, inverse)
    re.set(rowRe, y * nx)
    im.set(rowIm, y * nx)
  }
  const colRe = new Float64Array(ny)
  const colIm = new Float64Array(ny)
  for (let x = 0; x < nx; x++) {
    for (let y = 0; y < ny; y++) {
      colRe[y] = re[y * nx + x]!
      colIm[y] = im[y * nx + x]!
    }
    fft1d(colRe, colIm, inverse)
    for (let y = 0; y < ny; y++) {
      re[y * nx + x] = colRe[y]!
      im[y * nx + x] = colIm[y]!
    }
  }
}
