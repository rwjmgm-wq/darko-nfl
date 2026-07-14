#!/usr/bin/env Rscript
# Install required R packages for cfbfastR data collection

message("Installing R dependencies for cfbfastR...")
message(paste(rep("=", 80), collapse = ""))

# CRAN packages
cran_packages <- c("dplyr", "tidyr", "readr")

for (pkg in cran_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    message(sprintf("Installing %s...", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org/")
  } else {
    message(sprintf("%s: already installed", pkg))
  }
}

# cfbfastR from GitHub
if (!require("cfbfastR", quietly = TRUE)) {
  message("\nInstalling cfbfastR from GitHub...")
  if (!require("devtools", quietly = TRUE)) {
    install.packages("devtools", repos = "https://cloud.r-project.org/")
  }
  devtools::install_github("sportsdataverse/cfbfastR")
} else {
  message("cfbfastR: already installed")
}

message("\n")
message(paste(rep("=", 80), collapse = ""))
message("Installation complete!")
message(paste(rep("=", 80), collapse = ""))

# Test cfbfastR
message("\nTesting cfbfastR...")
tryCatch({
  library(cfbfastR)
  message("cfbfastR loaded successfully!")

  # Test with a small query
  message("\nTesting API access with sample query...")
  test_data <- cfbd_pbp_data(
    year = 2023,
    week = 1,
    team = "Alabama"
  )

  if (nrow(test_data) > 0) {
    message(sprintf("SUCCESS! Retrieved %d plays", nrow(test_data)))
    message("\nReady to collect college career data!")
  } else {
    message("WARNING: API returned no data")
  }

}, error = function(e) {
  message(sprintf("ERROR: %s", e$message))
  message("\nTroubleshooting:")
  message("  1. Check internet connection")
  message("  2. Verify CFBD API is accessible")
  message("  3. Try running manually: library(cfbfastR)")
})
