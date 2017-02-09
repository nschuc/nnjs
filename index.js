// @flow
import Tensor from './src/tensor.js';
import nj from 'numjs';

const randn = Tensor.randn;
const ones = Tensor.ones;

export default {
  Tensor,
  randn,
  ones
};
