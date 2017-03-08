import nj, { NdArray } from "numjs";

export default class Storage {
  _data: nj.array;

  constructor({ data, shape }: { data?: nj.array, shape?: Array<number> }) {
    if (data) {
      this._data = nj.array(data);
    } else {
      this._data = nj.zeros(shape, "float32");
    }
  }

  transpose(dims: Array<number> = [1, 0]) {
    return this.data.transpose(dims);
  }

  get data() {
    return this._data;
  }

  get size() {
    return this.data.shape;
  }

  add(t: Tensor | number) {
    const data = typeof t == "number" ? t : t.storage.data;
    return this.data.add(data);
  }

  add_(t: Tensor | number) {
    const data = typeof t == "number" ? t : t.storage.data;
    this.data.add(data, false);
    return this;
  }

  sub(t: Tensor | number) {
    const data = typeof t == "number" ? t : t.storage.data;
    return this.data.subtract(data);
  }

  select(dim: number, index: number) {
    if(this.size.length === 1) {
      return this.data.lo(index).hi(Math.min(this.size[0] - 1, index + 1));
    }

    let indices = new Array(this.data.shape.length);
    indices[dim] = index;
    return this.data.pick(...indices);
  }

  mm(t: Tensor | number) {
    const data = typeof t == "number" ? t : t.storage.data;
    return this.data.dot(data);
  }

  mul(t: Tensor | number) {
    const data = typeof t == "number" ? t : t.storage.data;
    return this.data.multiply(data);
  }

  pow(p: number) {
    return nj.power(this.data, p);
  }

  dist(t: Tensor) {
    return Math.sqrt(nj.power(this.data.subtract(t.storage.data), 2).sum());
  }

  norm() {
    return Math.sqrt(nj.power(this.data, 2).sum());
  }

  neg() {
    return this.data.negative();
  }

  zero_() {
    this._data.assign(0, false);
  }

  set_index_(index : number, t : Tensor) {
    let sub = this._data.lo(index).hi(Math.min(this.size[0] - 1, index + 1));
    sub.assign(t.list(), false)
  }

  toString() {
    return this.data.toString();
  }
}

