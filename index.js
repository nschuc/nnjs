// @flow
import Tensor from "./src/tensor.js";
import nj from "numjs";

const randn = Tensor.randn;
const ones = Tensor.ones;

const fromArray = (arr: Array<number>) => new Tensor({ data: nj.array(arr) });

export default { Tensor, randn, ones, fromArray }

