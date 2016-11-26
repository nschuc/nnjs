import test from 'ava';
import Graph from '../src/graph';

test('tensor shape gets set', t => {
  const G = new Graph()
  const t1 = G.variable([5, 3, 4])
  t.deepEqual(t1.getShape(), [5, 3, 4], 'does not get shape set');
})

test('tensor matmul computes correct shape', t => {
  const G = new Graph()

  const t1 = G.variable([50, 30])
  const t2 = G.variable([30, 25])
  const result = t1.mm(t2)

  t.deepEqual(result.getShape(), [50, 25], 'result of matrix mult is not right');
})

test('tensor matmul builds graph properly', t => {
  const G = new Graph()
  const W = G.variable([50, 30], 'W');
  const b = G.variable([50, 1], 'b');
  const x = G.input([30, 1], 'x');

  const result = W.mm(x).plus(b);

  t.is(G.nodes.size, 5, 'graph ops not set properly');
  t.deepEqual(result.getShape(), [50, 1], 'shape not computed correctly');
});


test('incompatible shapes throws an error', t => {
  const G = new Graph()
  const W = G.variable([50, 30], 'W');
  const b = G.variable([50, 1], 'b');

  t.throws(() => {
    const result = W.mm(b);
  });

  t.throws(() => {
    const result = W.mm(b);
  });
})
