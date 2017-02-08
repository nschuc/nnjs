import Tensor from '../src/tensor.js';
import { Variable } from '../src/autodiff';
import { closeEnough, matTestData, mmData } from './data.js';

describe('variable op mixins', () => {
  it('should have an add operation', () => {
    let t1 = new Variable(new Tensor({ shape: [5, 3] }));
    let t2 = new Variable(new Tensor({ shape: [5, 3] }));
    let t3 = t1.add(t2);
  });
})

describe('variable addition', () => {
  it('should have an add operation', () => {
    let v1 = new Variable(new Tensor({ shape: [5, 3] }));
    let v2 = new Variable(new Tensor({ shape: [5, 3] }));
    let v3 = v1.add(v2);
    const res = v3.data.numjs();
    expect(closeEnough(res, matTestData.a_plus_b)).toBe(true);
  });

  it('should have a matmul operation', () => {
    let W = new Variable(new Tensor({ data: mmData.W }));
    let x = new Variable(new Tensor({ data: mmData.x }));
    let y = W.matmul(x);
    expect(closeEnough(y.data.numjs(), mmData.y)).toBe(true);
  });
})
