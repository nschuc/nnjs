import Tensor from "../src/tensor.js";
import nn from "../";
import nj from "numjs";
import { closeEnough, matTestData, mmData, normData } from "./data.js";

describe("tensor destructuring", () => {
  it("should destructure 2 element array", () => {
    let t1 = new Tensor([1, 2]);
    let [x, y] = t1;
    expect(x).toEqual(1);
    expect(y).toEqual(2);
  });

  it("should destructure 3 element array", () => {
    let t1 = new Tensor([1, 2, 3]);
    let [x, y, z] = t1;
    expect(x).toEqual(1);
    expect(y).toEqual(2);
    expect(z).toEqual(3);
  });

  it("should destructure nested array", () => {
    let t1 = new Tensor([[1], [2], [3]]);
    let [x, y] = t1;
    expect(x).toEqual([1]);
    expect(y).toEqual([2]);
  });
});

describe("select subtensor", () => {
  it("should select proper subdimensions", () => {
    let t1 = new Tensor([
      [0, 1, 2, 3],
      [4, 5, 6, 7],
      [8, 9, 10, 11],
      [12, 13, 14, 15]
    ]);

    let h = t1.select(0, 0);
    let v = t1.select(1, 1);
    expect(h.list()).toEqual([0, 1, 2, 3]);
    expect(v.list()).toEqual([1, 5, 9, 13]);
  });
});

describe("index into tensor", () => {
  it("should index into 1-D tensor", () => {
    let t1 = new Tensor([1.5, 1.5]);
    let first = t1.index(0);
    let second = t1.index(1);
    expect(first.list()).toEqual([1.5]);
    expect(second.list()).toEqual([1.5]);
  });

  it("should index into 2-D tensor", () => {
    let t1 = new Tensor([
      [0, 1, 2, 3],
      [4, 5, 6, 7],
      [8, 9, 10, 11],
      [12, 13, 14, 15]
    ]);

    let h = t1.index(0).numjs();
    let v = t1.index(1).numjs();
    expect(h.tolist()).toEqual([0, 1, 2, 3]);
    expect(v.tolist()).toEqual([4, 5, 6, 7]);
  });
});

describe("tensor addition", () => {
  it("should create a third tensor with same shape", () => {
    let t1 = new Tensor(5, 3);
    let t2 = new Tensor(5, 3);
    let t3 = t1.add(t2);
    expect(t3.size).toEqual([5, 3]);
  });

  it("should add two tensors", () => {
    let t1 = new Tensor(matTestData.a);
    let t2 = new Tensor(matTestData.b);
    let t3 = t1.add(t2);
    const res = t3.numjs();
    expect(closeEnough(res, matTestData.a_plus_b)).toBe(true);
  });
});

describe("matrix multiplication", () => {
  it("should create a third tensor with proper shape", () => {
    let W = new Tensor(mmData.W);
    let x = new Tensor(mmData.x);
    let y = W.mm(x);
    expect(y.size).toEqual([3, 3]);
  });

  it("should multiply two matrices", () => {
    let W = new Tensor(mmData.W);
    let x = new Tensor(mmData.x);
    let y = W.mm(x);
    expect(closeEnough(y.numjs(), mmData.y)).toBe(true);
  });
});

describe("in-place zero", () => {
  it("should zero out all entries", () => {
    let x = Tensor.ones(3, 3);
    x.zero_()
    expect(x.list()).toEqual(Tensor.zeros(3,3).list());
  });
});

describe("norm", () => {
  it("should compute norm", () => {
    let x = new Tensor(normData.x);
    expect(x.norm()).toBeCloseTo(normData.norm);
  });
});
