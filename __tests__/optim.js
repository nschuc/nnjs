import Tensor from "../src/tensor.js";
import { Variable } from "../src/autodiff";
import { SGD } from "../src/optim";
import { closeEnough } from "./data.js";

const rosenbrock = (params) => {
  let [x, y] = params;
  let a = x.neg().add(1).pow(2).add(100);
  let b = y.sub(x.pow(2)).pow(2);
  return a.mul(b);
};

describe("SGD test", () => {
  it("2-D convex", () => {
    const soln = new Tensor([1, 1]);
    let params = new Variable(new Tensor([25, 1.5]));
    let opt = new SGD([params], { lr: 1e-3 });
    const fn = ([x, y]) => x.pow(2).add(y.pow(2));

    for (let i = 0; i < 2000; ++i) {
      opt.zero_grad();
      let loss = fn(params);
      loss.backward();
      opt.step();
    }

    let loss = fn(params);
    expect(loss.data.list()[0]).toBeLessThan(1);
  });
  it("rosenbrock", () => {
    const soln = new Variable(new Tensor([1, 1]));
    let params = new Variable(new Tensor([1.5, 1.5]));
    let opt = new SGD([params], { lr: 1e-4 });
    const initial_loss = params.dist(soln);

    for (let i = 0; i < 2000; ++i) {
      opt.zero_grad();
      let loss = rosenbrock(params);
      loss.backward();
      opt.step();
    }
    const loss = params.dist(soln);
    expect(loss.data).toBeLessThan(initial_loss.data);
  });
});
