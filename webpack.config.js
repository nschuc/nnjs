module.exports = {
  entry: './index.js',
  output: {
    path: './build',
    filename: 'index.bundle.js',
    libraryTarget: "var",
    library: "nn"
  },
  module: {
    loaders: [{
      exclude: /node_modules/,
      test: /\.js$/,
      loader: 'babel-loader',
      query: {
        presets: ['es2015']
      }
    },
    { test: /\.json$/, loader: 'json' },
    ]
  }
};
