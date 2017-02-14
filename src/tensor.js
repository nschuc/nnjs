//@flow
import nj, { NdArray } from "numjs";

class Storage {
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

  *[Symbol.iterator]() {
    for (let i = 0; i < this.data.shape[0]; i++) {
      yield this.data.slice([i, i + 1]);
    }
  }

  add(t: Tensor | number) {
    const data = typeof t == "number" ? t : t.storage.data;
    return this.data.add(data);
  }

  sub(t: Tensor | number) {
    const data = typeof t == "number" ? t : t.storage.data;
    return this.data.subtract(data);
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

  pow(p: number) {
    return nj.power(this.data, 2);
  }

  norm() {
    return Math.sqrt(nj.power(this.data, 2).sum());
  }

  neg() {
    return this.data.negative();
  }
}

export default class Tensor {
  storage: Storage;

  constructor(...args: any) {
    if (args.length == 1) {
      this.storage = new Storage({ data: args[0] });
    } else if (args.length > 1) {
      this.storage = new Storage({ shape: args });
    }
  }

  numjs() {
    return this.storage.data;
  }

  get T(): Tensor {
    const data = this.storage.transpose();
    return new Tensor(data);
  }

  get size(): Tensor {
    return this.storage.data.shape;
  }

  *[Symbol.iterator]() {
    yield* this.storage.data.tolist();
  }

  select(dim, index) {
    return new Tensor(this.storage.select(dim, index));
  }

  add(other: Tensor | number) {
    const data = this.storage.add(other);
    return new Tensor(data);
  }

  sub(other: Tensor | number) {
    const data = this.storage.sub(other);
    return new Tensor(data);
  }

  mm(other: Tensor) {
    const data = this.storage.mm(other);
    return new Tensor(data);
  }

  mul(other: number | Tensor) {
    const data = this.storage.mul(other);
    return new Tensor(data);
  }

  pow(other: number) {
    const data = this.storage.pow(other);
    return new Tensor(data);
  }

  norm() {
    return this.storage.norm();
  }

  neg() {
    const data = this.storage.neg();
    return new Tensor(data);
  }

  static ones(...shape: Array<number>) {
    return new Tensor(nj.ones(shape));
  }

  static randn(...shape: Array<number>) {
    return new Tensor(nj.random(shape));
  }
}
