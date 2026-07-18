/** Row-major dense matrix helpers for viz and bridging libraries. */

export type Matrix = {
  rows: number;
  cols: number;
  data: Float64Array;
};

export function mat(rows: number, cols: number, fill = 0): Matrix {
  return { rows, cols, data: new Float64Array(rows * cols).fill(fill) };
}

export function fromNested(a: number[][]): Matrix {
  const rows = a.length;
  const cols = a[0]?.length ?? 0;
  const m = mat(rows, cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) m.data[i * cols + j] = a[i][j];
  }
  return m;
}

export function toNested(m: Matrix): number[][] {
  const out: number[][] = [];
  for (let i = 0; i < m.rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < m.cols; j++) row.push(m.data[i * m.cols + j]);
    out.push(row);
  }
  return out;
}

export function get(m: Matrix, i: number, j: number): number {
  return m.data[i * m.cols + j];
}

export function set(m: Matrix, i: number, j: number, v: number): void {
  m.data[i * m.cols + j] = v;
}

export function copy(m: Matrix): Matrix {
  return { rows: m.rows, cols: m.cols, data: new Float64Array(m.data) };
}

export function transpose(m: Matrix): Matrix {
  const t = mat(m.cols, m.rows);
  for (let i = 0; i < m.rows; i++) {
    for (let j = 0; j < m.cols; j++) set(t, j, i, get(m, i, j));
  }
  return t;
}

export function matmul(a: Matrix, b: Matrix): Matrix {
  if (a.cols !== b.rows) {
    throw new Error(`matmul shape ${a.rows}x${a.cols} @ ${b.rows}x${b.cols}`);
  }
  const c = mat(a.rows, b.cols);
  for (let i = 0; i < a.rows; i++) {
    for (let k = 0; k < a.cols; k++) {
      const aik = get(a, i, k);
      for (let j = 0; j < b.cols; j++) {
        c.data[i * c.cols + j] += aik * get(b, k, j);
      }
    }
  }
  return c;
}

export function sub(a: Matrix, b: Matrix): Matrix {
  const c = copy(a);
  for (let i = 0; i < c.data.length; i++) c.data[i] -= b.data[i];
  return c;
}

export function frobeniusSq(m: Matrix): number {
  let s = 0;
  for (let i = 0; i < m.data.length; i++) s += m.data[i] * m.data[i];
  return s;
}

export function identity(n: number): Matrix {
  const I = mat(n, n);
  for (let i = 0; i < n; i++) set(I, i, i, 1);
  return I;
}

export function diag(values: number[]): Matrix {
  const n = values.length;
  const D = mat(n, n);
  for (let i = 0; i < n; i++) set(D, i, i, values[i]);
  return D;
}

/**
 * Mulberry32: deterministic PRNG so a displayed 32-bit seed reproduces a run.
 * Chosen because JS has no seedable built-in, the whole state is one integer
 * (fits in a status line / URL param), it passes gjrand over its full 2³²
 * period — plenty for demo inits — and it matches the generator already used
 * by scripts/gen-ringing-plot.mjs, so seeds mean the same thing everywhere.
 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randomNormal(
  rows: number,
  cols: number,
  scale = 1,
  rand: () => number = Math.random,
): Matrix {
  const m = mat(rows, cols);
  for (let i = 0; i < m.data.length; i++) {
    const u = Math.max(1e-12, rand());
    const v = rand();
    m.data[i] = scale * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  return m;
}

export function reconstruct(U: Matrix, sigma: number[], V: Matrix): Matrix {
  return matmul(matmul(U, diag(sigma)), transpose(V));
}

/**
 * Thin Q factor via modified Gram–Schmidt (Stiefel retraction).
 * Columns of the result are orthonormal: QᵀQ = I.
 */
export function thinQ(m: Matrix): Matrix {
  const Q = copy(m);
  const { rows: n, cols: k } = Q;
  for (let j = 0; j < k; j++) {
    let normSq = 0;
    for (let i = 0; i < n; i++) {
      const v = get(Q, i, j);
      normSq += v * v;
    }
    const inv = 1 / Math.sqrt(Math.max(normSq, 1e-30));
    for (let i = 0; i < n; i++) set(Q, i, j, get(Q, i, j) * inv);
    for (let ell = j + 1; ell < k; ell++) {
      let dot = 0;
      for (let i = 0; i < n; i++) dot += get(Q, i, j) * get(Q, i, ell);
      for (let i = 0; i < n; i++) {
        set(Q, i, ell, get(Q, i, ell) - dot * get(Q, i, j));
      }
    }
  }
  return Q;
}

/** Write a Matrix into a nested number[][] tensor buffer in place. */
export function writeIntoNested(data: unknown, m: Matrix): void {
  if (!Array.isArray(data)) {
    throw new Error("tensor data is not nested arrays");
  }
  for (let i = 0; i < m.rows; i++) {
    const row = data[i];
    if (!Array.isArray(row)) throw new Error("expected 2D nested tensor data");
    for (let j = 0; j < m.cols; j++) {
      (row as number[])[j] = get(m, i, j);
    }
  }
}

export function maxAbs(m: Matrix): number {
  let mabs = 0;
  for (let i = 0; i < m.data.length; i++) mabs = Math.max(mabs, Math.abs(m.data[i]));
  return mabs || 1;
}

export function fromTensorData(data: unknown, rows: number, cols: number): Matrix {
  const flat: number[] = [];
  const walk = (x: unknown) => {
    if (typeof x === "number") flat.push(x);
    else if (Array.isArray(x)) x.forEach(walk);
  };
  walk(data);
  if (flat.length !== rows * cols) {
    throw new Error(`expected ${rows * cols} values, got ${flat.length}`);
  }
  return { rows, cols, data: Float64Array.from(flat) };
}

export function takeColumns(m: Matrix, k: number): Matrix {
  const kk = Math.min(k, m.cols);
  const out = mat(m.rows, kk);
  for (let i = 0; i < m.rows; i++) {
    for (let j = 0; j < kk; j++) set(out, i, j, get(m, i, j));
  }
  return out;
}
