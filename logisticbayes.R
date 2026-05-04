suppressPackageStartupMessages({
  library(rstan)
  library(coda)
  library(ggplot2)
  library(ggmcmc)
})

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
set.seed(1234)

# ----------------------------
# Simulate weakly informed hierarchical logistic data
# ----------------------------
J <- 25
N_per_group <- sample(3:6, J, replace = TRUE)
N <- sum(N_per_group)
group <- rep(seq_len(J), times = N_per_group)
x <- rnorm(N, 0, 1)

alpha_true <- -0.6
beta_true <- 1.2
tau_true <- 1.4
b_true <- rnorm(J, 0, tau_true)

eta <- alpha_true + beta_true * x + b_true[group]
p <- plogis(eta)
y <- rbinom(N, 1, p)

stan_data <- list(
  N = N,
  J = J,
  y = y,
  x = x,
  group = group
)

# ----------------------------
# Stan models: centered vs non-centered
# ----------------------------
stan_centered <- "
data {
  int<lower=1> N;
  int<lower=1> J;
  array[N] int<lower=0,upper=1> y;
  vector[N] x;
  array[N] int<lower=1,upper=J> group;
}
parameters {
  real alpha;
  real beta;
  vector[J] b;
  real<lower=0> tau;
}
model {
  alpha ~ normal(0, 2);
  beta ~ normal(0, 2);
  tau ~ normal(0, 1);
  b ~ normal(0, tau);
  y ~ bernoulli_logit(alpha + beta * x + b[group]);
}
"

stan_ncp <- "
data {
  int<lower=1> N;
  int<lower=1> J;
  array[N] int<lower=0,upper=1> y;
  vector[N] x;
  array[N] int<lower=1,upper=J> group;
}
parameters {
  real alpha;
  real beta;
  vector[J] z;
  real<lower=0> tau;
}
transformed parameters {
  vector[J] b = tau * z;
}
model {
  alpha ~ normal(0, 2);
  beta ~ normal(0, 2);
  tau ~ normal(0, 1);
  z ~ normal(0, 1);
  y ~ bernoulli_logit(alpha + beta * x + b[group]);
}
"

# ----------------------------
# Utility diagnostics
# ----------------------------
count_divergences <- function(fit) {
  sp <- get_sampler_params(fit, inc_warmup = FALSE)
  per_chain <- sapply(sp, function(x) sum(x[, "divergent__"]))
  data.frame(
    chain = seq_along(per_chain),
    divergences = as.integer(per_chain)
  )
}

check_divergences <- function(fit) {
  d <- count_divergences(fit)
  total <- sum(d$divergences)
  cat("\nDivergences by chain:\n")
  print(d, row.names = FALSE)
  cat("Total divergences:", total, "\n")
  invisible(d)
}

check_treedepth <- function(fit) {
  sp <- get_sampler_params(fit, inc_warmup = FALSE)
  max_td <- get_num_upars(fit) # placeholder to avoid NULL note
  rm(max_td)
  td_hits <- sapply(sp, function(x) {
    # near-max treedepth is often a useful warning sign
    sum(x[, "treedepth__"] >= (max(x[, "treedepth__"]) - 1))
  })
  cat("\nNear-max treedepth hits by chain:\n")
  print(data.frame(chain = seq_along(td_hits), near_max_td_hits = td_hits), row.names = FALSE)
  invisible(td_hits)
}

ess_fraction <- function(fit) {
  s <- summary(fit)$summary
  post_warmup_draws <- fit@sim$n_save[1] - fit@sim$warmup2[1]
  total_draws_all_chains <- post_warmup_draws * fit@sim$chains
  n_eff <- s[, "n_eff"]
  out <- data.frame(
    parameter = rownames(s),
    n_eff = n_eff,
    total_post_warmup_draws = total_draws_all_chains,
    eff_fraction = n_eff / total_draws_all_chains
  )
  out
}

report_ess <- function(fit, top_n = 12) {
  e <- ess_fraction(fit)
  cat("\nESS fraction (first parameters):\n")
  print(head(e, top_n), row.names = FALSE)
  cat("Median ESS fraction:", round(median(e$eff_fraction, na.rm = TRUE), 3), "\n")
  invisible(e)
}

# ----------------------------
# Fit centered model with multiple adapt_delta values
# ----------------------------
mod_c <- stan_model(model_code = stan_centered)
mod_ncp <- stan_model(model_code = stan_ncp)

fit_c_adapt_08 <- sampling(
  mod_c, data = stan_data, iter = 2000, warmup = 1000, chains = 4, seed = 101,
  control = list(adapt_delta = 0.8)
)

fit_c_adapt_05 <- sampling(
  mod_c, data = stan_data, iter = 2000, warmup = 1000, chains = 4, seed = 102,
  control = list(adapt_delta = 0.5)
)

fit_c_adapt_095 <- sampling(
  mod_c, data = stan_data, iter = 2000, warmup = 1000, chains = 4, seed = 103,
  control = list(adapt_delta = 0.95)
)

fit_ncp_adapt_08 <- sampling(
  mod_ncp, data = stan_data, iter = 2000, warmup = 1000, chains = 4, seed = 201,
  control = list(adapt_delta = 0.8)
)

# ----------------------------
# Print diagnostics requested in the prompt
# ----------------------------
cat("\n=== Centered (adapt_delta = 0.5) ===\n")
check_divergences(fit_c_adapt_05)
check_treedepth(fit_c_adapt_05)
ess_c05 <- report_ess(fit_c_adapt_05)

cat("\n=== Centered (adapt_delta = 0.8) ===\n")
check_divergences(fit_c_adapt_08)
check_treedepth(fit_c_adapt_08)
ess_c08 <- report_ess(fit_c_adapt_08)

cat("\n=== Centered (adapt_delta = 0.95) ===\n")
check_divergences(fit_c_adapt_095)
check_treedepth(fit_c_adapt_095)
ess_c095 <- report_ess(fit_c_adapt_095)

cat("\n=== Non-centered (adapt_delta = 0.8) ===\n")
check_divergences(fit_ncp_adapt_08)
check_treedepth(fit_ncp_adapt_08)
ess_ncp <- report_ess(fit_ncp_adapt_08)

# ----------------------------
# Visual diagnostics from the paper suggestions
# ----------------------------
# pairs plot for funnel-relevant parameters
pairs(fit_c_adapt_08, pars = c("tau", "b[1]", "b[2]", "b[3]"))
pairs(fit_ncp_adapt_08, pars = c("tau", "z[1]", "z[2]", "z[3]"))

# ggmcmc traceplots (convert stanfit -> mcmc.list first)
samp_c <- ggs(As.mcmc.list(fit_c_adapt_08))
samp_ncp <- ggs(As.mcmc.list(fit_ncp_adapt_08))

print(
  ggs_traceplot(
    samp_c,
    family = "alpha|beta|tau"
  ) +
    ggtitle("Centered traceplot") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
)

print(
  ggs_traceplot(
    samp_ncp,
    family = "alpha|beta|tau"
  ) +
    ggtitle("Non-centered traceplot") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
)

# ----------------------------
# Simple side-by-side summary table for reporting
# ----------------------------
summ_row <- function(name, fit, ess_tbl) {
  d <- count_divergences(fit)
  data.frame(
    model = name,
    divergences_total = sum(d$divergences),
    median_ess_fraction = median(ess_tbl$eff_fraction, na.rm = TRUE)
  )
}

diag_summary <- rbind(
  summ_row("centered_adapt_0.5", fit_c_adapt_05, ess_c05),
  summ_row("centered_adapt_0.8", fit_c_adapt_08, ess_c08),
  summ_row("centered_adapt_0.95", fit_c_adapt_095, ess_c095),
  summ_row("noncentered_adapt_0.8", fit_ncp_adapt_08, ess_ncp)
)

cat("\n=== Diagnostic Summary ===\n")
print(diag_summary, row.names = FALSE)
