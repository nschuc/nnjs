// @flow
import Tensor from "./src/tensor.js";
import optim from "./src/optim";

const randn = Tensor.randn;
const ones = Tensor.ones;

export default { Tensor, randn, ones, optim };
