import Tensor from "../src/tensor.js";
import { Variable } from "../src/autodiff";
import { closeEnough } from "./data.js";
import * as testData from "./data.js";
import nn from "../";
import nj from "numjs";

describe("variable op mixins", () => {
  it("should have an add operation", () => {
    let t1 = new Variable(new Tensor(5, 3));
    let t2 = new Variable(new Tensor(5, 3));
    let t3 = t1.add(t2);
  });
});

describe("variable addition", () => {
  it("should have an add operation", () => {
    let t1 = new Tensor(testData.matrix.a);
    let t2 = new Tensor(testData.matrix.b);

    let v1 = new Variable(t1);
    let v2 = new Variable(t2);
    let v3 = v1.add(v2);
    const res = v3.data.numjs();
    expect(closeEnough(res, testData.matrix.a_plus_b)).toBe(true);
  });

  it("should have a matmul operation", () => {
    let W = new Variable(new Tensor(testData.matmul.W));
    let x = new Variable(new Tensor(testData.matmul.x));
    let y = W.matmul(x);
    expect(closeEnough(y.data.numjs(), testData.matmul.y)).toBe(true);
  });
});

describe("variable gradient", () => {
  it("should have grad set when backward is called", () => {
    let x = new Variable(Tensor.ones(3));
    let y = x.mul(2);
    while (y.data.norm() < 1000) {
      y = y.mul(2);
    }

    const gradients = new Tensor([0.1, 1, 0.0001]);

    y.backward(gradients);

    const grad = x.grad.list();
    const expectedGrad = [102.4, 1024, 0.1024];
    expect(grad).toEqual(expectedGrad);
  });
});
