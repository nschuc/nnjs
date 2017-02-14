import Tensor from "../src/tensor.js";
import { Variable } from "../src/autodiff";
import nn from "../";

const rosenbrock = (x, y) => {
  let a = x.neg().add(1).pow(2).add(100);
  let b = y.sub(x.pow(2)).pow(2);
  return a.mul(b);
};

describe("variable gradient", () => {
  it("should have grad set when backward is called", () => {
    const x = new Variable(new Tensor([1.5]));
    const y = new Variable(new Tensor([1.5]));
    const loss = rosenbrock(x, y);
    console.log(loss);
    //expect(closeEnough(grad, expectedGrad)).toBe(true);
  });
});
