import * as esprima from 'esprima';

export const closeEnough = (a, b, eps=1e-3) => {
  for(let i = 0; i < a.data.length; i++) {
    if(Math.abs(a.data[i] - b.data[i]) > eps) {
      return false;
    }
  }
  return true;
}


export const getId = (layers, type) => {
  let idx = 1;
  let lcType = type.toLocaleLowerCase();
  while(layers[`${lcType}_${idx}`]) idx++;
  return `${lcType}_${idx}`
}

/**
 * Traverses ast applying visitors to nodes along the
 * way.
 *
 * args:
 *  node: ast node to traverse
 *  visitors: object of visitors with each property
 *    mapping from a ES node name to a function
 *    taking (node, state) params
 *  state: initial state that will get passed to
 *    all the visitors
 */
export const traverse = (node, visitors, state = {}) => {
  for(const key in node) {
    if(node.hasOwnProperty(key)) {
      const child = node[key];
      if(child && typeof child === 'object') {
        if(Array.isArray(child)) {
          child.forEach((node) => traverse(node, visitors, state));
        }
        else {
          traverse(child, visitors, state);
        }
        for(const vKey in visitors) {
          if(child.type === vKey) {
            const ret = visitors[vKey](child, state);
            if(ret && typeof ret === 'object') {
              node[key] = ret;
            }
          }
        }
      }
    }
  }
}

/**
 * Returns a ES AST node for a callsite with
 * the identifiers specified in ...args
 */
export const createCallSite = (callee, ...args) => {
  return {
    "type": "CallExpression",
    "callee": {
      "type": "Identifier",
      "name": callee
    },
    "arguments": [ ...args, ]
  }
}

/**
 * Builds ast for op and removes program wrapper
 */
const getOpAst = (op) => {
  if(typeof op === 'object') return op;
  const str = op.toString();
  const ast = esprima.parse(str);
  return ast.body[0]
}

/**
 * Builds asts for the functions in ops and injects
 * them to the beginning of the ast
 */
export const injectOps = (ast, ops) => {
  let functionBody = Array.isArray(ast.body[0].body.body) ? ast.body[0].body.body : [ ast.body[0].body.body ];
  let opAsts = Object.keys(ops).map(name => getOpAst(ops[name]));
  ast.body[0].body.body = [...opAsts, ...functionBody];
}
