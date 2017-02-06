data {
  int num_data;
  int num_basis;
  vector[num_data] Y;
  vector[num_data] X;
  matrix[num_basis, num_data] B;
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
  a = a_raw*tau; 
  Y_hat = a0*X + to_vector(a*B);
}

model {
  a_raw ~ normal(0, 1);
  tau ~ cauchy(0, 1);
  sigma ~ cauchy(0, 1);
  Y ~ normal(Y_hat, sigma);
}