import Tensor from '../src/tensor.js'
import nj from 'numjs';
import { closeEnough, matTestData, mmData } from './data.js';

describe('tensor addition', () => {
  it('should create a third tensor with same shape', () => {
    let t1 = new Tensor({ shape: [5, 3] });
    let t2 = new Tensor({ shape: [5, 3] });
    let t3 = t1.add(t2);
    expect(t3.shape).toEqual([5, 3]);
  });

  it('should add two tensors', () => {
    let t1 = new Tensor({ shape: [5, 3] });
    let t2 = new Tensor({ shape: [5, 3]});
    let t3 = t1.add(t2);
    const res = t3.numjs();
    expect(closeEnough(res, matTestData.a_plus_b)).toBe(true);
  });
})

describe('matrix multiplication', () => {
  it('should create a third tensor with proper shape', () => {
    let W = new Tensor({ data: mmData.W });
    let x = new Tensor({ data: mmData.x });
    let y = W.mm(x);
    expect(y.shape).toEqual([3, 3]);
  });

  it('should multiply two matrices', () => {
    let W = new Tensor({ data: mmData.W });
    let x = new Tensor({ data: mmData.x });
    let y = W.mm(x);
    expect(closeEnough(y.numjs(), mmData.y)).toBe(true);
  });
})
