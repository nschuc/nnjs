// @flow
import Tensor from './src/tensor.js';
import nj from 'numjs';

const randn = (...shape : Array<number> ) => {
  return new Tensor({ data: nj.random(shape) });
}

export {
  Tensor,
  randn
};
