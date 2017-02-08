import nj from 'numjs';

export const closeEnough = (a, b, eps=1e-3) => {
  a = a.tolist();
  b = b.tolist();
  for(let i = 0; i < a.length; i++) {
    if(Math.abs(a[i] - b[i]) > eps) {
      return false;
    }
  }
  return true;
}

export const matTestData = {
  a: nj.array([[0, -1], [1, 0], [4.7, 12]]),
  b: nj.array([[1, 2.3], [1, 1], [3, 4]]),
  a_plus_b: nj.array([[1, 1.3], [2, 1], [7.7, 16]]),
  a_sub_b: nj.array([[-1, -3.3], [0, -1], [1.7, 8]]),
  a_pow_2: nj.array([[0, 1], [1, 0], [22.09, 144]]),
}

export const mmData = {
  W: nj.array([[0, -1], [1, 0], [4.7, 12]]),
  x: nj.array([[1, 2.3, 1], [1, 3, 4]]),
  y: nj.array([[-1, -3, -4], [1, 2.3, 1], [16.7, 46.81, 52.7]])
}

describe('test data', () => {
  it('should exist', () => {
  });
})
