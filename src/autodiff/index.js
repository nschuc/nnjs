import Variable from './variable.js';
import Tensor from '../tensor.js';
import *  as ops from './ops.js';

for(let k of Object.keys(ops)) {
  Variable._registerOp(k.toLowerCase(), ops[k]);
}

export {
  Variable
}
