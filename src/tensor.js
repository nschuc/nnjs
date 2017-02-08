//@flow
import nj, { NdArray } from 'numjs';

class Storage {
  data: nj.array;

  constructor({ data, shape } : { data? : nj.array, shape? : Array<number> }) {
    this.data = data || nj.zeros(shape, 'float32');
  }

  add(t : Storage) {
    return this.data.add(t.data);
  }

  mm(t : Storage) {
    return this.data.dot(t.data);
  }
}

export default class Tensor {
  shape: Array<number>;
  storage: Storage;

  constructor({ data, shape = [] } : { data? : nj.array, shape? : Array<number> }) {
    if(data) {
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

  add(t : Tensor) {
    const data = this.storage.add(t.storage);
    return new Tensor({ data });
  }

  mm(t : Tensor) {
    const data = this.storage.mm(t.storage);
    return new Tensor({ data });
  }
}
