module.exports = {
  entry: './index.js',
  output: {
    path: './build',
    filename: 'index.bundle.js',
    libraryTarget: "var",
    library: "nn"
  },
  resolve: {
    // Add `.ts` and `.tsx` as a resolvable extension.
    extensions: ['', '.webpack.js', '.web.js', '.ts', '.tsx', '.js']
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
    { test: /\.ts$/, loader: 'ts-loader' },
    { test: /\.json$/, loader: 'json' },
    ]
  }
};
