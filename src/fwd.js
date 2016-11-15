import * as esprima from 'esprima';
import * as escodegen from 'escodegen';

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

const createCallSite = (callee, ...args) => {
  return {
    "type": "CallExpression",
    "callee": {
      "type": "Identifier",
      "name": callee
    },
    "arguments": [ ...args, ]
  }
}

const injectOps = (ast, ops) => {
  let functionBody = Array.isArray(ast.body[0].body.body) ? ast.body[0].body.body : [ ast.body[0].body.body ];
  let opAsts = Object.keys(ops).map(name => getOpAst(ops[name]));
  ast.body[0].body.body = [...opAsts, ...functionBody];
}

const getIdentifiers = (node, identifiers = new Set()) => {
  for(const key in node) {
    if(node.hasOwnProperty(key)) {
      const child = node[key];
      if(child && typeof child === 'object') {
        if(Array.isArray(child)) {
          child.forEach((node) => getIdentifiers(node, identifiers));
        }
        else {
          getIdentifiers(child, identifiers);
        }
        if(child.type === 'Identifier') {
          identifiers.add(child.name, identifiers)
        }
      }
    }
  }
  return identifiers
}

const traverse = (node, opNames) => {
  for(const key in node) {
    if(node.hasOwnProperty(key)) {
      const child = node[key];
      if(child && typeof child === 'object') {
        if(Array.isArray(child)) {
          child.forEach((node) => traverse(node, opNames));
        }
        else {
          traverse(child, opNames);
        }
        if(child.type === 'BinaryExpression') {
          const opNode = createCallSite(opNames[child.operator], child.left, child.right);
          node[key] = opNode;
        }
      }
    }
  }
}

const fwd = (fn) => {
  if( typeof fn !== 'function' ) {
    throw 'nice try buddy'
  }

  // COnstruct ast
  const fnString = fn.toString();
  var ast = esprima.parse(fnString);

  // inject operator overloads
  let identifiers = getIdentifiers(ast);

  traverse(ast, {
    '*': 'mult',
    '+': 'add'
  })

  injectOps(ast, opFuncs);
  const generatedCode = escodegen.generate(ast);
  return Function('return ' + generatedCode)()
}

export default fwd 
