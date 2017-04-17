import nj from 'numjs';
import allclose from 'allclose';

export const closeEnough = (a, b, eps=1e-3) => {
  a = a.tolist();
  b = b.tolist();
  return allclose(a, b, eps);
}

export const matrix = {
  a: nj.array([[0, -1], [1, 0], [4.7, 12]]),
  b: nj.array([[1, 2.3], [1, 1], [3, 4]]),
  a_plus_b: nj.array([[1, 1.3], [2, 1], [7.7, 16]]),
  a_sub_b: nj.array([[-1, -3.3], [0, -1], [1.7, 8]]),
  a_pow_2: nj.array([[0, 1], [1, 0], [22.09, 144]]),
  a_pow_grad: nj.array([[0.0, -2.0], [2.0, 0.0], [9.4, 24.0]]),
}

export const matmul = {
  W: nj.array([[0, -1], [1, 0], [4.7, 12]]),
  x: nj.array([[1, 2.3, 1], [1, 3, 4]]),
  y: nj.array([[-1, -3, -4], [1, 2.3, 1], [16.7, 46.81, 52.7]])
}

export const norm = {
  x: nj.array([[0, -1], [1, 0], [4.7, 12]]),
  y: 12.96
}

export const sigmoid = {
  x: nj.array([ 1.72926998,  0.54779869,  0.56997693, -2.02737784,  1.82751369]),
  y: nj.array([ 0.84931904,  0.63362473,  0.63875782,  0.11635826,  0.86146528]),
  grad: nj.array([ 0.12797621,  0.23214443,  0.23074627,  0.10281901,  0.11934286])
    
}


describe('test data', () => {
  it('should exist', () => {
  });
})
