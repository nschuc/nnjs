//@flow
import Storage from './storage';
import nj from 'numjs';

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

  list() {
    return this.storage.data.tolist();
  }

  get T(): Tensor {
    const data = this.storage.transpose();
    return new Tensor(data);
  }

  get size(): Tensor {
    return this.storage.size;
  }

  *[Symbol.iterator]() {
    yield* this.storage.data.tolist();
  }

  select(dim, index) {
    return new Tensor(this.storage.select(dim, index));
  }

  index(index) {
    return this.select(0, index);
  }

  add(other: Tensor | number) {
    const data = this.storage.add(other);
    return new Tensor(data);
  }

  add_(other: Tensor | number) {
    this.storage.add_(other);
    return this;
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

  dist(other : Tensor) {
    return this.storage.dist(other);
  }

  neg() {
    const data = this.storage.neg();
    return new Tensor(data);
  }

  zero_() {
    this.storage.zero_();
    return this;
  }

  set_index_(index: number, t : Tensor) {
    this.storage.set_index_(index, t);
    return this;
  }


  static zeros(...shape: Array<number>) {
    return new Tensor(nj.zeros(shape));
  }

  static ones(...shape: Array<number>) {
    return new Tensor(nj.ones(shape));
  }

  static randn(...shape: Array<number>) {
    return new Tensor(nj.random(shape));
  }
}

Tensor.prototype.toString = function() {
  return `${this.storage.toString()}\n\t[Tensor with size ${this.size.join('x')}]`;
};

Tensor.prototype.inspect = Tensor.prototype.toString;
