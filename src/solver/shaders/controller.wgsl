// Step-size controller. Reduces the per-workgroup error sums written by the final combine stage,
// decides whether the step is accepted and picks the next step size, entirely on the GPU.
// The error estimate is the difference between the 4th-order result and the (2nd-order) third
// stage, so the local error scales like h³ (order k = 3).
//
// The new step size comes from a PI controller (Gustafsson): with e = err/tol and e₋ the same ratio
// of the previous accepted step,  factor = safety · e^(−0.7/k) · e₋^(0.4/k).  The memory term damps the
// oscillation of the plain controller (factor = e^(−1/k)) between accepted and rejected steps.
// After a rejection the plain exponent −1/k is used instead, to shrink faster.
//
// ctl layout: 0 h, 1 t, 2 accepted (1/0), 3 error, 4 accepted steps, 5 rejected steps,
// 6 error of the previous accepted step (0: none yet), 7 unused.

struct Params {
  count: u32, // number of partial sums
  coefWorkgroups: u32, // workgroups of the coefficient kernel when h changes
  tol: f32,
  hmin: f32,
  hmax: f32,
  safety: f32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> partials: array<vec2<f32>>; // (Σ|Δ|², Σ|u|²) per workgroup
@group(0) @binding(2) var<storage, read_write> ctl: array<f32>;
@group(0) @binding(3) var<storage, read_write> args: array<u32>; // indirect dispatch of the coefficient kernel

var<workgroup> sums: array<vec2<f32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  var acc = vec2<f32>(0.0);
  for (var i = li; i < p.count; i += 256u) {
    acc += partials[i];
  }
  sums[li] = acc;
  workgroupBarrier();
  var s = 128u;
  loop {
    if (s == 0u) {
      break;
    }
    if (li < s) {
      sums[li] += sums[li + s];
    }
    workgroupBarrier();
    s = s >> 1u;
  }
  if (li != 0u) {
    return;
  }

  let h = ctl[0];
  let err = sqrt(sums[0].x / (sums[0].y + 1e-30));
  let ok = err <= p.tol; // false for NaN
  let accept = ok || h <= p.hmin * 1.0001;
  var factor = 0.2;
  if (err < 1e30) {
    let e = max(err / p.tol, 1e-3);
    var previous = e; // without history the memory term just weakens the response
    if (accept && ctl[6] > 0.0) {
      previous = max(ctl[6] / p.tol, 1e-3);
    }
    factor = clamp(p.safety * pow(e, -0.7 / 3.0) * pow(previous, 0.4 / 3.0), 0.2, 2.0);
    if (!ok) {
      // a rejected step is retried with the plain controller, which shrinks faster
      factor = clamp(p.safety * pow(e, -1.0 / 3.0), 0.2, 2.0);
    }
  }
  if (!ok) {
    factor = min(factor, 0.9);
  }
  var newH = h;
  // accepted steps whose factor is close to 1 keep h (saves recomputing the coefficients)
  if (!accept || factor >= 1.19 || factor < 0.9) {
    newH = clamp(h * factor, p.hmin, p.hmax);
  }
  if (accept) {
    ctl[1] = ctl[1] + h;
    ctl[4] = ctl[4] + 1.0;
    ctl[6] = select(0.0, err, err < 1e30);
  } else {
    ctl[5] = ctl[5] + 1.0;
  }
  ctl[0] = newH;
  ctl[2] = select(0.0, 1.0, accept);
  ctl[3] = err;
  args[0] = select(0u, p.coefWorkgroups, newH != h);
}
