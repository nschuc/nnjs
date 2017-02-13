import Tensor from "../src/tensor.js";
import nn from "../";
import nj from "numjs";
import { closeEnough, matTestData, mmData, normData } from "./data.js";

describe("tensor destructuring", () => {
  it("should destructure 2 element array", () => {
    let t1 = nn.fromArray([ 1, 2 ]);
    let [ x, y ] = t1;
    expect(x).toEqual(1);
    expect(y).toEqual(2);
  });

  it("should destructure 3 element array", () => {
    let t1 = nn.fromArray([ 1, 2, 3 ]);
    let [ x, y, z ] = t1;
    expect(x).toEqual(1);
    expect(y).toEqual(2);
    expect(z).toEqual(3);
  });

  it("should destructure nested array", () => {
    let t1 = nn.fromArray([ [ 1 ], [ 2 ], [ 3 ] ]);
    let [ x, y ] = t1;
    expect(x).toEqual([ 1 ]);
    expect(y).toEqual([ 2 ]);
  });
});

describe("select subtensor", () => {
  it("should select proper subdimensions", () => {
    let t1 = nn.fromArray([
      [ 0, 1, 2, 3 ],
      [ 4, 5, 6, 7 ],
      [ 8, 9, 10, 11 ],
      [ 12, 13, 14, 15 ]
    ]);

    let h = t1.select(0, 0).numjs();
    let v = t1.select(1, 1).numjs();
    expect(h.tolist()).toEqual([ 0, 1, 2, 3 ]);
    expect(v.tolist()).toEqual([ 1, 5, 9, 13 ]);
  });
});

describe("tensor addition", () => {
  it("should create a third tensor with same shape", () => {
    let t1 = new Tensor({ shape: [ 5, 3 ] });
    let t2 = new Tensor({ shape: [ 5, 3 ] });
    let t3 = t1.add(t2);
    expect(t3.shape).toEqual([ 5, 3 ]);
  });

  it("should add two tensors", () => {
    let t1 = new Tensor({ data: matTestData.a });
    let t2 = new Tensor({ data: matTestData.b });
    let t3 = t1.add(t2);
    const res = t3.numjs();
    expect(closeEnough(res, matTestData.a_plus_b)).toBe(true);
  });
});

describe("matrix multiplication", () => {
  it("should create a third tensor with proper shape", () => {
    let W = new Tensor({ data: mmData.W });
    let x = new Tensor({ data: mmData.x });
    let y = W.mm(x);
    expect(y.shape).toEqual([ 3, 3 ]);
  });

  it("should multiply two matrices", () => {
    let W = new Tensor({ data: mmData.W });
    let x = new Tensor({ data: mmData.x });
    let y = W.mm(x);
    expect(closeEnough(y.numjs(), mmData.y)).toBe(true);
  });
});

describe("norm", () => {
  it("should compute norm", () => {
    let x = new Tensor({ data: normData.x });
    expect(x.norm()).toBeCloseTo(normData.norm);
  });
});

