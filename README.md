# nnjs
Small library for building Neural Networks in Javascript. 

Pre-pre-alpha.

Example inspired by [Deep Learning with PyTorch: A 60-minute Blitz](https://github.com/pytorch/tutorials/blob/master/Deep%20Learning%20with%20PyTorch.ipynb)
```javascript
let x = new Variable(nn.ones(3));
let y = x.mul(2);

while (y.data.norm() < 1000) {
  y = y.mul(2);
}

y.backward(nn.fromArray([ 0.1, 1, 0.0001 ]));

const grad = x.grad.numjs();
const expectedGrad = nj.array([ 102.4, 1024, 0.1024 ]);
```
