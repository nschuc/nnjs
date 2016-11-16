import * as esprima from 'esprima';
import * as escodegen from 'escodegen';

import { traverse, createCallSite, injectOps } from './utils';

const getOpAst = (op) => {
  const str = op.toString();
  const ast = esprima.parse(str);
  return ast.body[0]
}

const opFuncs = { 
  'wrapOp': (a, b, op) => {
    a = typeof a === 'number' ? [a, 0] : a;
    b = typeof b === 'number' ? [b, 0] : b;
    return op(a, b);
  },
  'mult': (a, b) => wrapOp(a, b, (a, b) => {
    return [
      a[0] * b[0],
      a[1] * b[0] + a[0] * b[1]
    ]}),
  'add': (a, b) => {
    a = typeof a === 'number' ? [a, 0] : a;
    b = typeof b === 'number' ? [b, 0] : b;
    return [
      a[0] + b[0],
      a[1] + b[1]
    ]},
}

const getIdentifiers = (ast) => {
  let identifiers = new Set();
  traverse(ast, {
    Identifier: (node, state) => {
      state.identifiers.add(node.name, identifiers)
    }
  }, {
    identifiers
  });
  return identifiers;
}

const fwd = (fn) => {
  if( typeof fn !== 'function' ) {
    throw 'nice try buddy'
  }

  // Construct ast
  const fnString = fn.toString();
  var ast = esprima.parse(fnString);

  // inject operator overloads

  traverse(ast, {
    BinaryExpression: (node, state) => {
      const opNames = {
        '*': 'mult',
        '+': 'add'
      }
      const opNode = createCallSite(opNames[node.operator], node.left, node.right);
      return opNode;
    }
  })

  injectOps(ast, opFuncs);
  const generatedCode = escodegen.generate(ast);
  return Function('return ' + generatedCode)()
}

export default fwd 
