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

export function randomNormal(rows: number, cols: number, scale = 1): Matrix {
  const m = mat(rows, cols);
  for (let i = 0; i < m.data.length; i++) {
    const u = Math.max(1e-12, Math.random());
    const v = Math.random();
    m.data[i] = scale * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  return m;
}

export function reconstruct(U: Matrix, sigma: number[], V: Matrix): Matrix {
  return matmul(matmul(U, diag(sigma)), transpose(V));
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
