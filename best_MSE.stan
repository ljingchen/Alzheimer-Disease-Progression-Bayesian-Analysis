data {
  int<lower=0> N_train;   // number of data items in the train dataset
  int<lower=0> P;   // number of predictors
  matrix[N_train, P] x_train;   // predictor matrix
  int<lower = 0, upper = 1> y_train[N_train];
  int<lower = 1> N_test;
  matrix[N_test, P] x_test;
}
parameters {
  vector[P] beta;
  real alpha;
}
model {
  beta ~ normal(0, 100);
  alpha ~ normal(0, 10);
  y_train ~ bernoulli_logit(alpha + x_train*beta);
}
generated quantities {
  vector[N_test] y_test;
  for(i in 1:N_test) {
    y_test[i] = bernoulli_rng(inv_logit(x_test[i]*beta+alpha));
  }
}



