#!/usr/bin/env Rscript
# Collect College Career Data using cfbfastR
#
# Alternative to CFBD API - uses cfbfastR which may have better auth handling

library(cfbfastR)
library(dplyr)
library(tidyr)
library(readr)

# Read QB list
message("Loading QB list...")
qb_list <- read_csv("data/processed/qb_list_expanded.csv")
message(sprintf("Loaded %d QBs", nrow(qb_list)))

# Create output directory
output_dir <- "data/processed/full_college_careers"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Function to get all seasons for a QB
get_qb_seasons <- function(draft_year, college) {
  final_season <- draft_year - 1
  # Look back 5 years (covers redshirt seniors)
  potential_seasons <- (final_season - 4):final_season
  return(potential_seasons)
}

# Function to collect QB career data
collect_qb_career <- function(player_name, draft_year, college) {
  message(sprintf("Processing: %s (%s, %d)", player_name, college, draft_year))

  all_plays <- data.frame()
  seasons <- get_qb_seasons(draft_year, college)

  for (season in seasons) {
    tryCatch({
      message(sprintf("  Fetching %d...", season))

      # Get play-by-play data for this season
      pbp <- cfbd_pbp_data(
        year = season,
        team = college
      )

      if (nrow(pbp) > 0) {
        # Filter to passing plays by this QB
        qb_plays <- pbp %>%
          filter(
            play_type == "Pass",
            !is.na(passer_player_name),
            grepl(player_name, passer_player_name, ignore.case = TRUE)
          ) %>%
          mutate(
            season = season,
            player_name = player_name,
            college = college,
            draft_year = draft_year
          ) %>%
          select(
            season, player_name, college, draft_year,
            down, distance, yards_gained, play_type,
            EPA, scoring, yards_to_goal,
            period, clock_minutes, clock_seconds
          )

        if (nrow(qb_plays) > 0) {
          message(sprintf("    Found %d plays", nrow(qb_plays)))
          all_plays <- bind_rows(all_plays, qb_plays)
        }
      }

      # Rate limiting
      Sys.sleep(0.5)

    }, error = function(e) {
      message(sprintf("    Error: %s", e$message))
    })
  }

  return(all_plays)
}

# Process each QB
message("\nCollecting career data...")
message(paste(rep("=", 80), collapse = ""))

career_summary <- data.frame()
success_count <- 0
fail_count <- 0

for (i in 1:nrow(qb_list)) {
  row <- qb_list[i, ]

  career_data <- collect_qb_career(
    row$player_name,
    row$draft_year,
    row$college
  )

  if (nrow(career_data) > 0) {
    # Save this QB's career
    filename <- sprintf("%s_%d.csv",
                       gsub(" ", "_", row$player_name),
                       row$draft_year)
    filepath <- file.path(output_dir, filename)
    write_csv(career_data, filepath)

    # Update summary
    seasons <- unique(career_data$season)
    career_summary <- bind_rows(career_summary, data.frame(
      player_name = row$player_name,
      draft_year = row$draft_year,
      college = row$college,
      seasons_played = length(seasons),
      first_season = min(seasons),
      final_season = max(seasons),
      total_plays = nrow(career_data),
      file = filename
    ))

    success_count <- success_count + 1
    message(sprintf("  [%d/%d] %s: SUCCESS (%d plays)",
                   i, nrow(qb_list), row$player_name, nrow(career_data)))
  } else {
    fail_count <- fail_count + 1
    message(sprintf("  [%d/%d] %s: FAILED (no plays found)",
                   i, nrow(qb_list), row$player_name))
  }
}

# Save career summary
if (nrow(career_summary) > 0) {
  summary_path <- file.path(output_dir, "career_summary.csv")
  write_csv(career_summary, summary_path)

  message("\n")
  message(paste(rep("=", 80), collapse = ""))
  message("COLLECTION COMPLETE")
  message(paste(rep("=", 80), collapse = ""))
  message(sprintf("\nSuccessfully collected: %d QBs", success_count))
  message(sprintf("Failed: %d QBs", fail_count))
  message(sprintf("Average seasons per QB: %.1f", mean(career_summary$seasons_played)))
  message(sprintf("Total plays collected: %s", format(sum(career_summary$total_plays), big.mark = ",")))
  message(sprintf("\nData saved to: %s", output_dir))
} else {
  message("\nERROR: No data collected")
}
