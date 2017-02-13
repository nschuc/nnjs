//@flow
import nj, { NdArray } from "numjs";

class Storage {
  _data: nj.array;

  constructor({ data, shape }: { data?: nj.array, shape?: Array<number> }) {
    this._data = data || nj.zeros(shape, "float32");
  }

  transpose(dims: Array<number> = [ 1, 0 ]) {
    return this.data.transpose(dims);
  }

  get data() {
    return this._data;
  }

  *[Symbol.iterator]() {
    for (let i = 0; i < this.data.shape[0]; i++) {
      yield this.data.slice([ i, i + 1 ]);
    }
  }

  add(t: Tensor | number) {
    const data = typeof t == "number" ? t : t.storage.data;
    return this.data.add(data);
  }

  select(dim: number, index: number) {
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

  norm() {
    return Math.sqrt(nj.power(this.data, 2).sum());
  }
}

export default class Tensor {
  shape: Array<number>;
  storage: Storage;

  constructor(
    { data, shape = [] }: { data?: nj.array, shape?: Array<number> }
  ) {
    if (data) {
      shape = data.shape;
    }
    this.shape = shape;
    this.storage = new Storage({ data, shape });
  }

  shape() {
    return this.shape;
  }

  numjs() {
    return this.storage.data;
  }

  get T(): Tensor {
    const data = this.storage.transpose();
    return new Tensor({ data });
  }

  *[Symbol.iterator]() {
    yield* this.storage.data.tolist();
  }

  select(dim, index) {
    return new Tensor({ data: this.storage.select(dim, index) });
  }

  add(other: Tensor | number) {
    const data = this.storage.add(other);
    return new Tensor({ data });
  }

  mm(other: Tensor) {
    const data = this.storage.mm(other);
    return new Tensor({ data });
  }

  mul(other: number | Tensor) {
    const data = this.storage.mul(other);
    return new Tensor({ data });
  }

  norm() {
    return this.storage.norm();
  }

  static ones(...shape: Array<number>) {
    return new Tensor({ data: nj.ones(shape) });
  }

  static randn(...shape: Array<number>) {
    return new Tensor({ data: nj.random(shape) });
  }
}

