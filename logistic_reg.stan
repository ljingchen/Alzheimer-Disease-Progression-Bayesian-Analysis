data {
  int<lower=0> N;   // number of data items
  int<lower=0> P;   // number of predictors
  matrix[N, P] x;   // predictor matrix
  int<lower=0,upper=1> y[N];  // binary outcome
}

parameters {
  vector[P] beta;
  real alpha;
}

model {
  beta ~ normal(0, 100);
  alpha ~ normal(0, 10);
  y ~ bernoulli_logit(alpha+x * beta);
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = bernoulli_logit_lpmf(y[n] | x[n] * beta+alpha);
  }
}



