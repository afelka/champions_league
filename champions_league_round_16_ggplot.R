library(ggplot2)
library(dplyr)
library(sf)
library(rnaturalearth)
library(av)

# Europe + Cyprus + Turkey and its neighbors towards Russia
countries <- ne_countries(scale = "medium", returnclass = "sf") %>%
  filter(
    (continent == "Europe" | 
       name %in% c("Turkey", "Cyprus", "Georgia", "Armenia", "Azerbaijan")) &
      name != "Iceland"   # drop Iceland to put summary table there
  )

# UK subdivisions
uk_regions <- ne_states(country = "United Kingdom", returnclass = "sf") %>%
  filter(geonunit %in% c("England", "Scotland")) %>%
  mutate(name = geonunit)  # rename geonunit -> name

# Remove UK polygon
countries_no_uk <- countries %>% filter(name != "United Kingdom")

# Combine
map_polygons <- bind_rows(countries_no_uk, uk_regions)

# Clean data
teams <- teams_per_year_per_country %>%
  mutate(country = recode(country, "Czech Republic" = "Czechia")) %>%
  mutate(count_f = factor(count, levels = 1:6))

# Define colors
colors_7 <- c("grey80", "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02")

# Create output folder
if(!dir.exists("season_maps")) dir.create("season_maps")

seasons <- unique(teams$season)

for(season_i in seasons){
  season_data <- teams %>% filter(season == season_i)
  
  # Join with polygons
  map_df <- map_polygons %>%
    left_join(season_data, by = c("name" = "country")) %>%
    mutate(count_f = factor(ifelse(is.na(count_f), "0", as.character(count_f)), 
                            levels = c("0", "1", "2", "3", "4", "5", "6")))
  
  # Summary table
  summary_table <- season_data %>%
    group_by(country) %>%
    summarise(count = sum(count)) %>%
    arrange(desc(count)) %>%
    mutate(label = paste0(country, " ", count)) %>%
    pull(label) %>%
    paste(collapse = "\n")
  
  # Add header
  summary_text <- paste0("Teams per association\n", summary_table)
  
  # Coordinates for annotation (northern Europe)
  annotation_lon <- -20
  annotation_lat <- 65
  
  # plot per season
  p <- ggplot() +
    geom_sf(data = map_df, aes(fill = count_f), color = "white", size = 0.2) +
    scale_fill_manual(
      values = colors_7,
      drop = FALSE,   # NOT ddrop
      name = "Teams"
    ) +
    coord_sf(xlim = c(-20, 45), ylim = c(35, 65)) +
    labs(title = paste0(season_i, " UEFA Champions League Round 16")) +
    theme_void() +
    theme(
      plot.background = element_rect(fill = "grey90", color = NA),
      panel.background = element_rect(fill = "grey90", color = NA),
      plot.title = element_text(hjust = 0.5, size = 16)
    ) +
    annotate("text",
             x = annotation_lon,
             y = annotation_lat,
             label = summary_text,
             hjust = 0,
             vjust = 1,
             size = 3.5,
             color = "black",
             lineheight = 0.9
    )
  
  # Save each season map
  ggsave(
    filename = sprintf("season_maps/%s.png", gsub("–","_",season_i)),
    plot = p,
    width = 8, height = 6, dpi = 300
  )
}

# list filenames 
filenames <- list.files(
  "season_maps",
  pattern = "\\.png$",    # all png files
  full.names = TRUE
)

# Repeat each frame 2 times to make it 2 seconds per frame
filenames_long <- rep(filenames, each = 2)

# using the animation method found here : https://stackoverflow.com/a/73376411/10710995
av::av_encode_video(filenames_long, framerate = 1,
                    output = "champions_league_round_16.mp4")
