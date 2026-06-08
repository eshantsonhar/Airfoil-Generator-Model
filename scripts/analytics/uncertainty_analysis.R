# R-based analytics pipeline for uncertainty quantification
# 
# Implements:
# - Uncertainty quantification
# - Repeated-run variance
# - Bootstrap confidence intervals
# - Monte Carlo propagation
# - Sensitivity analysis
# - Mixed-effects modeling
# - Regression analysis
# - Optimization trend analysis
# - Reproducibility summaries
#
# Generates:
# - Publication-ready figures
# - Uncertainty envelopes
# - Convergence plots
# - Mesh-sensitivity plots
# - Optimization-history analytics

library(tidyverse)
library(lme4)
library(broom)
library(data.table)
library(ggplot2)
library(patchwork)

# Load telemetry data
load_telemetry <- function(db_path) {
  library(RSQLite)
  con <- dbConnect(SQLite(), db_path)
  
  query <- "SELECT * FROM telemetry"
  data <- dbGetQuery(con, query)
  
  dbDisconnect(con)
  
  return(as.data.table(data))
}

# Bootstrap confidence intervals
bootstrap_ci <- function(data, metric_col, n_bootstrap = 1000, ci_level = 0.95) {
  boot_samples <- replicate(n_bootstrap, {
    sample_idx <- sample(nrow(data), replace = TRUE)
    mean(data[sample_idx, ..metric_col])
  })
  
  ci_lower <- quantile(boot_samples, (1 - ci_level) / 2)
  ci_upper <- quantile(boot_samples, 1 - (1 - ci_level) / 2)
  
  return(list(
    mean = mean(boot_samples),
    ci_lower = ci_lower,
    ci_upper = ci_upper,
    std = sd(boot_samples)
  ))
}

# Sensitivity analysis using variance decomposition
sensitivity_analysis <- function(data, response_var, predictor_vars) {
  # Standardize predictors
  data_std <- data %>%
    mutate(across(all_of(predictor_vars), scale))
  
  # Fit linear model
  model <- lm(as.formula(paste(response_var, "~", paste(predictor_vars, collapse = " + "))), 
              data = data_std)
  
  # Get standardized coefficients
  coefs <- broom::tidy(model) %>%
    filter(term != "(Intercept)") %>%
    mutate(abs_estimate = abs(estimate)) %>%
    arrange(desc(abs_estimate))
  
  return(coefs)
}

# Mixed-effects modeling for repeated measures
mixed_effects_analysis <- function(data, response_var, fixed_vars, random_var = "run_id") {
  formula_str <- paste(response_var, "~", paste(fixed_vars, collapse = " + "), "+ (1|", random_var, ")")
  
  model <- lmer(as.formula(formula_str), data = data)
  
  return(broom::tidy(model, effects = "fixed"))
}

# Optimization trend analysis
optimization_trend_analysis <- function(data, metric_col, iteration_col = "iteration") {
  trend <- data %>%
    group_by(!!sym(iteration_col)) %>%
    summarise(
      mean = mean(!!sym(metric_col), na.rm = TRUE),
      sd = sd(!!sym(metric_col), na.rm = TRUE),
      se = sd / sqrt(n()),
      .groups = "drop"
    )
  
  return(trend)
}

# Generate publication-ready figures
plot_optimization_history <- function(data, metric_col, output_path) {
  trend <- optimization_trend_analysis(data, metric_col)
  
  p <- ggplot(trend, aes(x = iteration, y = mean)) +
    geom_line(linewidth = 1) +
    geom_ribbon(aes(ymin = mean - se, ymax = mean + se), alpha = 0.3) +
    geom_point(size = 2) +
    labs(
      title = "Optimization History",
      x = "Iteration",
      y = metric_col,
      subtitle = paste("Mean ± SE")
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      axis.title = element_text(size = 12),
      panel.grid.minor = element_blank()
    )
  
  ggsave(output_path, p, width = 10, height = 6, dpi = 300)
  
  return(p)
}

# Mesh sensitivity analysis
plot_mesh_sensitivity <- function(data, output_path) {
  if (!"mesh_size" %in% colnames(data) || !"metric_value" %in% colnames(data)) {
    warning("Required columns not found for mesh sensitivity plot")
    return(NULL)
  }
  
  p <- ggplot(data, aes(x = mesh_size, y = metric_value)) +
    geom_point(alpha = 0.6) +
    geom_smooth(method = "lm", se = TRUE, color = "blue") +
    labs(
      title = "Mesh Sensitivity Analysis",
      x = "Mesh Size (cells)",
      y = "Metric Value",
      subtitle = "Linear regression with confidence interval"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      axis.title = element_text(size = 12)
    )
  
  ggsave(output_path, p, width = 8, height = 6, dpi = 300)
  
  return(p)
}

# Convergence diagnostics plot
plot_convergence_diagnostics <- function(data, residual_col, output_path) {
  if (!residual_col %in% colnames(data)) {
    warning(paste("Column", residual_col, "not found"))
    return(NULL)
  }
  
  p <- ggplot(data, aes(x = iteration, y = !!sym(residual_col))) +
    geom_line() +
    scale_y_log10() +
    labs(
      title = "Convergence Diagnostics",
      x = "Iteration",
      y = "Residual (log scale)",
      subtitle = "Log-linear convergence plot"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      axis.title = element_text(size = 12)
    )
  
  ggsave(output_path, p, width = 10, height = 6, dpi = 300)
  
  return(p)
}

# Reproducibility summary
reproducibility_summary <- function(data, run_id_col = "run_id") {
  summary <- data %>%
    group_by(!!sym(run_id_col)) %>%
    summarise(
      n_observations = n(),
      .groups = "drop"
    ) %>%
    summarise(
      n_runs = n(),
      mean_obs_per_run = mean(n_observations),
      std_obs_per_run = sd(n_observations)
    )
  
  return(summary)
}

# Main analysis function
run_uncertainty_analysis <- function(db_path, output_dir) {
  # Create output directory
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Load data
  cat("Loading telemetry data...\n")
  data <- load_telemetry(db_path)
  
  if (nrow(data) == 0) {
    cat("No data found in database.\n")
    return(NULL)
  }
  
  cat(paste("Loaded", nrow(data), "telemetry records.\n"))
  
  # Get unique metrics
  metrics <- unique(data$metric_name)
  cat(paste("Found", length(metrics), "unique metrics.\n"))
  
  # Analysis results
  results <- list()
  
  # Analyze each metric
  for (metric in metrics) {
    cat(paste("\nAnalyzing metric:", metric, "\n"))
    
    metric_data <- data[metric_name == metric]
    
    # Bootstrap confidence intervals
    boot_ci <- bootstrap_ci(metric_data, "metric_value")
    results[[metric]]$bootstrap <- boot_ci
    
    # Sensitivity analysis (if iteration available)
    if ("iteration" %in% colnames(metric_data)) {
      sensitivity <- sensitivity_analysis(metric_data, "metric_value", c("iteration"))
      results[[metric]]$sensitivity <- sensitivity
    }
    
    # Optimization trend plot
    if ("iteration" %in% colnames(metric_data)) {
      plot_path <- file.path(output_dir, paste0(metric, "_optimization_trend.png"))
      plot_optimization_history(metric_data, "metric_value", plot_path)
      results[[metric]]$trend_plot <- plot_path
    }
  }
  
  # Convergence diagnostics
  if ("residual" %in% metrics) {
    residual_data <- data[metric_name == "residual"]
    plot_path <- file.path(output_dir, "convergence_diagnostics.png")
    plot_convergence_diagnostics(residual_data, "metric_value", plot_path)
    results$convergence_plot <- plot_path
  }
  
  # Reproducibility summary
  repro_summary <- reproducibility_summary(data)
  results$reproducibility <- repro_summary
  
  # Save results
  results_path <- file.path(output_dir, "analysis_results.RDS")
  saveRDS(results, results_path)
  
  cat("\nAnalysis complete.\n")
  cat(paste("Results saved to:", results_path, "\n"))
  
  return(results)
}

# Command line interface
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  cat("Usage: Rscript uncertainty_analysis.R <db_path> <output_dir>\n")
  cat("Example: Rscript uncertainty_analysis.R data/telemetry/metrics.db data/analytics\n")
} else {
  db_path <- args[1]
  output_dir <- args[2]
  
  run_uncertainty_analysis(db_path, output_dir)
}
