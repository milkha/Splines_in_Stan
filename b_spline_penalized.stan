functions {
  vector build_b_spline(real[] t, real[] ext_knots, int ind, int order);
  vector build_b_spline(real[] t, real[] ext_knots, int ind, int order) {
    vector[size(t)] b_spline;
    vector[size(t)] w1 = rep_vector(0, size(t));
    vector[size(t)] w2 = rep_vector(0, size(t));
    if (order==1)
      for (i in 1:size(t))
        b_spline[i] = (ext_knots[ind] <= t[i]) && (t[i] < ext_knots[ind+1]); 
    else {
      if (ext_knots[ind] != ext_knots[ind+order-1])
        w1 = (to_vector(t) - rep_vector(ext_knots[ind], size(t))) / 
             (ext_knots[ind+order-1] - ext_knots[ind]);
      if (ext_knots[ind+1] != ext_knots[ind+order])
        w2 = 1 - (to_vector(t) - rep_vector(ext_knots[ind+1], size(t))) / 
                 (ext_knots[ind+order] - ext_knots[ind+1]);
      b_spline = w1 .* build_b_spline(t, ext_knots, ind, order-1) + 
                 w2 .* build_b_spline(t, ext_knots, ind+1, order-1);
    }
    return b_spline;
  }
}

data {
  int num_data;
  int num_knots;
  vector[num_knots] knots;
  int spline_degree;
  real Y[num_data];
  real X[num_data];
}

transformed data {
  int num_basis = num_knots + spline_degree - 1; 
  matrix[num_basis, num_data] B;
  vector[spline_degree + num_knots] ext_knots_temp;
  vector[2*spline_degree + num_knots] ext_knots;
  ext_knots_temp = append_row(rep_vector(knots[1], spline_degree), knots);
  ext_knots = append_row(ext_knots_temp, rep_vector(knots[num_knots], spline_degree));
  for (ind in 1:num_basis)
    B[ind,:] = to_row_vector(build_b_spline(X, to_array_1d(ext_knots), ind, spline_degree + 1));
  B[num_knots + spline_degree - 1, num_data] = 1;

}
parameters {
  row_vector[num_basis] a_raw;
  real a0;
  real<lower=0> sigma;
  real<lower=0> tau;
}

transformed parameters {
  row_vector[num_basis] a;
  vector[num_data] Y_hat;
  a[1] = a_raw[1];
  for (i in 2:num_basis)
    a[i] = a[i-1] + a_raw[i]*tau; 
  Y_hat = a0*to_vector(X) + to_vector(a*B);
}

model {
  a_raw ~ normal(0, 1);
  tau ~ normal(0, 1);
  sigma ~ cauchy(0, 1);
  Y ~ normal(Y_hat, sigma);
}