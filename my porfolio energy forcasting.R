# Machine Learning

# Library --------

library(lightgbm)
library(xgboost)

# Tidymodels

library(tidymodels)         # Machine learning R packages
library(bonsai)             # Tidymodels interface to lightgbm
library(modeltime)          # Time series in tidymodels
library(modeltime.ensemble) # Ensembles

# Core
library(timetk)
library(tidyverse)
library(tidyquant)
library(readxl)
library(plotly)
library(workflowsets)       # Useful to manage multiple workflows
library(modeltime.gluonts)  # Deep Learning For Time Series

# Import data ----

raw_data <-  read_excel("data/energy_production.xlsx") |> mutate(date = as.Date(Date)) |>
             filter_by_time(.start_date = "2010-01-01") |> select(-Date)
raw_data

# Time series analysis ----

raw_data |>
     tk_summary_diagnostics()

raw_data |> count(MSN)
raw_data |> count(Description)
raw_data |> count(Unit)

# Examine the hierarchy
# Total Fossil Fuels Production (Sum of fossil fuel sources)
# -- Coal Production
# -- Crude Oil Production
# -- Natural Gas (Dry) Production
# -- Natural Gas Plant Liquids Production
# -- Nuclear Energy:

# --    Nuclear Electric Power Production

# Total Renewable Energy Production (Sum of renewable energy sources)

# -- Geothermal Energy Production
# -- Hydroelectric Power Production
# -- Solar Energy Production
# -- Wind Energy Production


# Total Primary Energy Production (Sum of all energy sources)

data <-  raw_data |>  mutate(energy_production_category =
                             case_when (
                                 str_detect(Description, "Coal|Oil|Gas|Nuclear")  ~ "Fossil Fuels Production",
                                 str_detect(Description, "Geo|Hydro|Solar|Wind|Biomass") ~ "Renewable Energy Production",
                                        TRUE  ~  'Double check' ) ) |> select(date, Value, energy_production_category,Description) |>
                                filter(energy_production_category != "Double check")


data

data |> distinct(date) |> pull()
data |> count(energy_production_category)
data |> filter(energy_production_category == "Double check") |> count(Description)

monthly_all_data <- data |> pad_by_time(.by = "month") |>
                    summarise_by_time(.by = "month", value = sum(Value))

monthly_all_data |> distinct(date) |> pull()

monthly_all_data |>
     plot_time_series(
         date, value,
         .facet_ncol = 1,
         .trelliscope = FALSE,
         # .interactive = F,
         .smooth = F
     )

# imputation
# consider imputing the data as average or other

monthly_data_level_1 <- data |> group_by(energy_production_category) |> pad_by_time(date,.by = "month", .pad_value=0) |>
                        mutate_by_time(.by = "month",
                                       value = sum(Value))

monthly_data_level_1 |>
    group_by(energy_production_category) |>
    plot_time_series(
        date, value,
        .facet_ncol = 3,
        .trelliscope = TRUE,
         .interactive = TRUE,
        .smooth = F
    )

# I will skip level 1  forecast at this stage

monthly_data_level_2 <- data |> group_by(Description) |> pad_by_time(date,.by = "month",.pad_value=0) |>
                        summarise_by_time(.by = "month",
                                          value = sum(Value))


monthly_data_level_2  |>
    group_by(Description) |>
    plot_time_series(
        date, value,
        .facet_ncol = 2,
        .trelliscope = TRUE,
        .interactive = TRUE,
        .smooth = F
    )

# Before imputation

monthly_data_level_2 |> group_by(Description) |>
                      mutate(value_na = if_else(value == 0 , NA,value)) |>
                      mutate(value_imputed = ts_impute_vec(value_na) ) |>
                      # pivot_longer(cols = c(value, value_imputed), names_to = "imputation", values_to = "value") |>
                      plot_time_series(date, value,
                                       .facet_ncol = 3, .interactive = TRUE)

# After imputation

monthly_data_level_2 |> group_by(Description) |>
                        mutate(value_na = if_else(value == 0 , NA,value)) |>
                        mutate(value_imputed = ts_impute_vec(value_na) ) |>
                        plot_time_series(date, value_imputed,
                                         .facet_ncol = 3, .interactive = TRUE)


monthly_data_level_2 <- monthly_data_level_2 |> group_by(Description) |>
                        mutate(value_na = if_else(value == 0 , NA,value)) |>
                        mutate(value = ts_impute_vec(value_na) ) |>
                        select(-value_na) |> ungroup()

monthly_data_level_2 |> group_by(Description) |>
                        plot_time_series(date, value,
                                       .facet_ncol = 3, .interactive = TRUE)

# - Detecting Trend and Seasonal Cycles - all data aggregated
# Seasonal decomposition

monthly_data_level_2 |> group_by(Description) |>
                        plot_stl_diagnostics(date, value,
                          #.frequency = "1 month",
                          #.trend = "1 year"
                          )

# - Detecting Trend and Seasonal Cycles - level 1 data aggregated
# Seasonal decomposition

monthly_data_level_1 |> group_by(energy_production_category) |>
    plot_stl_diagnostics(date, value,
                         #.frequency = "1 month",
                         #.trend = "1 year"
    )

# Before imputation

monthly_data_level_1 |> group_by(energy_production_category) |>
    mutate(value_na = if_else(value == 0 , NA,value)) |>
    mutate(value_imputed = ts_impute_vec(value_na) ) |>
    # pivot_longer(cols = c(value, value_imputed), names_to = "imputation", values_to = "value") |>
    plot_time_series(date, value,
                     .facet_ncol = 2, .interactive = TRUE)

# After imputation

monthly_data_level_1 |> group_by(energy_production_category) |>
    mutate(value_na = if_else(value == 0 , NA,value)) |>
    mutate(value_imputed = ts_impute_vec(value_na) ) |>
    plot_time_series(date, value_imputed,
                     .facet_ncol = 2, .interactive = TRUE)

monthly_data_level_1 <- monthly_data_level_1 |> group_by(energy_production_category) |>
                        mutate(value_na = if_else(value == 0 , NA,value)) |>
                        mutate(value = ts_impute_vec(value_na) ) |>
                        select(-value_na) |> ungroup()

monthly_data_level_1 |> group_by(energy_production_category) |> plot_time_series(date, value,
                                       .facet_ncol = 2, .interactive = TRUE)

?plot_seasonal_diagnostics

monthly_data_level_1 |> group_by(energy_production_category) |>
    plot_seasonal_diagnostics(
        .date_var = date,
        .value    = value,
        .title = "Seasonal Diagnostics",
        .feature_set = c("auto"),
        .geom        = "boxplot"
    )


# - Detecting Lagged Features

monthly_data_level_1 |> group_by(energy_production_category) |>
    plot_acf_diagnostics(date, value, .lags = 100)

# Detecting  Anomalies

monthly_data_level_1 |> group_by(energy_production_category) |>
    plot_anomaly_diagnostics(date, value)


# - Detecting Trend and Seasonal Cycles - level 2 data aggregated
# Seasonal decomposition

monthly_data_level_2 |> group_by(Description) |>
    filter(Description %in% c("Coal Production", "Crude Oil Production", "Natural Gas (Dry) Production"," Nuclear Electric Power Production")) |>
    plot_stl_diagnostics(date, value,
                         #.frequency = "1 month",
                         #.trend = "1 year"
    )


# to much items for visuals

monthly_data_level_2 |>
    filter(Description %in% c("Coal Production", "Crude Oil Production", "Natural Gas (Dry) Production")) |>
    #filer(Description %in% c("Geothermal Energy Production","")) |>
    plot_seasonal_diagnostics(
        .date_var = date,
        .value    = value,
        .title = "Seasonal Diagnostics",
        .feature_set = c("auto"),
        .geom        = "boxplot"
    )


monthly_data_level_2 |>
    filter(Description %in% c("Coal Production", "Crude Oil Production", "Natural Gas (Dry) Production","Nuclear Electric Power Production")) |>
    plot_acf_diagnostics(date, value, .lags = 100)

monthly_data_level_2 |>
    filter(Description %in% c("Geothermal Energy Production", "Hydroelectric Power Production", "Solar Energy Production","Wind Energy Production")) |>
    plot_acf_diagnostics(date, value, .lags = 100)

monthly_data_level_2 |>
    tk_summary_diagnostics()

 # GLOBAL MODEL------

#https://business-science.github.io/modeltime/articles/recursive-forecasting.html#recursive-forecasting-with-panel-models

## Data prep -----

monthly_data_level_2 <-  monthly_data_level_2 |> rename(id = Description)
monthly_data_level_2 |> glimpse()

monthly_data_level_2 |> group_by(id) |> distinct(date)

FORECAST_HORIZON <- as.integer(24)

data_extended_tbl <- monthly_data_level_2 |>
                    # Extend
                    group_by(id) |>
                    future_frame(date, .length_out = FORECAST_HORIZON, .bind_data = TRUE) |>
                    tk_augment_lags(value, .lags = FORECAST_HORIZON, .names = "long_lag") |>
                    tk_augment_slidify(
                        .value   = long_lag,
                        .f       = ~mean(.x, na.rm=TRUE),
                        .period  = c(0.5*FORECAST_HORIZON , FORECAST_HORIZON, FORECAST_HORIZON*1.5),
                        .align   = "center",
                        .partial = TRUE
                        )  |> ungroup()

data_extended_tbl |> glimpse()
data_extended_tbl |> group_by(id) |> filter(id == "Coal Production") |> plot_time_series(date, value)

monthly_data_level_2|>
    tk_summary_diagnostics()

lag_roll_transformer_grouped <- function(data) {
                                data |>
                                    group_by(id) |>
                                    tk_augment_lags(value, .lags = 1:FORECAST_HORIZON) |>
                                    tk_augment_slidify(
                                        .value   = contains("lag12"),
                                        .f       = ~mean(.x, na.rm = T),
                                        .period  = c(12),
                                        .partial = TRUE
                                    )   |> ungroup()
                            }

data_lagged_tbl <-  data_extended_tbl |>
                    group_by(id) |>
                    lag_roll_transformer_grouped() |>
                    # ***
                    # FIX 1: REMOVE DROP NA
                    # Issue: Causes future data to be removed because of
                    # missing values in lags (which are ok)
                    #drop_na(-value) |>
                    # END FIX
                    ungroup()


data_lagged_tbl |> glimpse()

# * Data Prepared & Data Future

data_prepared_tbl  <- data_lagged_tbl |>
                      drop_na()
data_prepared_tbl

data_future_tbl <- data_lagged_tbl |>
                   filter(is.na(value))

data_future_tbl |> distinct(date)

#MACHINE LEARNING

#TIME SPLIT

resamples <- data_prepared_tbl |>
             time_series_split(date, assess = 24, cumulative = TRUE)

resamples |>
    tk_time_series_cv_plan() |>
    plot_time_series_cv_plan(date, value)


train_tbl <- training(resamples) |> group_by(id) |> mutate(value = ts_clean_vec(value, period = 12))  |>
             ungroup()

train_tbl |>  group_by(id) |>  plot_time_series(date, value,.facet_ncol = 2,
                                                .trelliscope = TRUE)

test_tbl <- testing(resamples)
test_tbl |> distinct(date)

##Recipes ----

# ** Lag Recipe

recipe_spec_date <- recipe(value ~ ., data = train_tbl |> select(date,value))
recipe_spec_date |> prep() |> juice() |> glimpse()

recipe_spec_1 <- recipe(value ~ ., data = train_tbl) |> step_rm(starts_with("lag"))
recipe_spec_1 |> prep() |> juice() |> glimpse()

recipe_spec_lag <- recipe(value ~ ., data = train_tbl) |>
    step_dummy(all_nominal()) |>
    step_rm(date) |>
    step_zv(all_predictors())

recipe_spec_lag |> prep() |> juice() |> glimpse()

# ** Calendar + Long-Term Rolling Recipe

recipe_spec_calendar <- recipe(
                            value ~ .,
                            data = train_tbl |> select(-contains("lag"), contains("roll"))) |>
                            step_timeseries_signature(date) |>
                            step_dummy(all_nominal(), one_hot = TRUE) |>
                            step_normalize(date_index.num, starts_with("date_year")) |>
                            step_rm(date) |>
                            step_zv(all_predictors())


recipe_spec_calendar |> prep() |> juice() |> glimpse()

# ** Hybrid Short-Term Lag + Calendar Recipe

recipe_spec_hybrid <- recipe(value ~ ., data = train_tbl) |>
    step_timeseries_signature(date) |>
    step_dummy(all_nominal(), one_hot = TRUE) |>
    step_normalize(date_index.num, starts_with("date_year")) |>
    step_rm(date) |>
    step_zv(all_predictors())

recipe_spec_hybrid |> prep() |> juice() |> glimpse()


## Baseline Algoritms ----
# - Used to make simple window models (mean, median, seasonal naive)

# Mean by ID

model_mean_fit_6 <- window_reg(id = "id", window_size = 6) |>
                    set_engine("window_function", window_function = mean) |>
                    fit(value ~ ., data =train_tbl)

model_mean_fit_6


model_mean_fit_12 <- window_reg(id = "id", window_size = 12) |>
                     set_engine("window_function", window_function = mean) |>
                     fit(value ~ ., data =train_tbl)

model_mean_fit_12

# Seasonal NAIVE by ID

model_snaive_fit <- naive_reg(seasonal_period = 12, id = "id") |>
                    set_engine("snaive") |>
                    fit(value ~ ., data = train_tbl)

baseline_models <- modeltime_table(
                        model_mean_fit_6,
                        model_mean_fit_12,
                        model_snaive_fit
                    )

baseline_models

# Tuning the Window

window_grid_tbl <- tibble(window_size = 1:12) |>
                    create_model_grid(
                        f_model_spec  = window_reg,
                        id            = "id",
                        engine_name   = "window_function",
                        engine_params = list(window_function = ~ median(.)
                        )
                        )

window_model_list <- window_grid_tbl |>
                     pull(.models) |>
                     map(~ fit(., value ~ .,train_tbl))

baseline_window_tuned <- window_model_list |>
                         as_modeltime_table()

baseline_window_tuned


calibration_baseline_models_tbl  <- combine_modeltime_tables(
                                        baseline_window_tuned,
                                        baseline_models
                                            ) |>
                                        modeltime_calibrate(train_tbl, id = "id")

calibration_baseline_models_tbl

# * Elastic Net Penalized Regression

model_spec_glmnet <- linear_reg(penalty = 200,mixture = 0.99) |>
                     set_engine("glmnet")

# * Workflows

wflw_fit_glmnet_lag <-  workflow() |>
                        add_model(model_spec_glmnet) |>
                        add_recipe(recipe_spec_lag) |>
                        fit(train_tbl) |>
                        recursive(
                            id         = "id",
                            transform  = lag_roll_transformer_grouped,
                            train_tail = panel_tail(train_tbl, id, FORECAST_HORIZON)
                        )

wflw_fit_glmnet_calendar <- workflow() |>
    add_model(model_spec_glmnet) |>
    add_recipe(recipe_spec_calendar) |>
    fit(train_tbl)

wflw_fit_glmnet_hybrid <- workflow() |>
    add_model(model_spec_glmnet) |>
    add_recipe(recipe_spec_hybrid) |>
    fit(train_tbl) |>
    recursive(
        id = "id",
        transform = lag_roll_transformer_grouped,
        train_tail = panel_tail(train_tbl, id, FORECAST_HORIZON)
    )

# * Calibration Results

calibration_glmnet_tbl <- modeltime_table(
                            wflw_fit_glmnet_lag,
                            wflw_fit_glmnet_calendar,
                            wflw_fit_glmnet_hybrid
                                )  |>
                            modeltime_calibrate(test_tbl, id = "id")

calibration_glmnet_tbl |> modeltime_accuracy(acc_by_id = FALSE)
calibration_glmnet_tbl |> modeltime_accuracy(acc_by_id = TRUE)

# * Classical algorithms

model_spec_ets  <- exp_smoothing() |>
                   set_engine("ets")

wflw_fit_ets_recursive <-   workflow() |>
                            add_model(model_spec_ets) |>
                            add_recipe(recipe_spec_date) |>
                            fit(train_tbl) |>
                            recursive(
                                id         = "id",
                                transform  = lag_roll_transformer_grouped,
                                train_tail = panel_tail(train_tbl, id, FORECAST_HORIZON)
                            )

wflw_fit_ets_basic <-   workflow() |>
                        add_model(model_spec_ets) |>
                        add_recipe(recipe_spec_date) |>
                        fit(train_tbl)
# theta model

model_spec_theta <-  exp_smoothing() |>
                     set_engine("ets")

wflw_fit_theta_recursive <-   workflow() |>
    add_model(model_spec_theta) |>
    add_recipe(recipe_spec_date) |>
    fit(train_tbl) |>
    recursive(
        id         = "id",
        transform  = lag_roll_transformer_grouped,
        train_tail = panel_tail(train_tbl, id, FORECAST_HORIZON)
    )

wflw_fit_theta_basic <- workflow() |>
                        add_model(model_spec_theta) |>
                        add_recipe(recipe_spec_date) |>
                        fit(train_tbl)

# tbats model

model_spec_tbats <-  seasonal_reg(
                            seasonal_period_1 = 7,
                            seasonal_period_2 = 30,
                            seasonal_period_3 = 365
                            ) |>
                    set_engine("tbats")



wflw_fit_tbats_recursive <- workflow() |>
                            add_model(model_spec_tbats) |>
                            add_recipe(recipe_spec_date) |>
                            fit(train_tbl) |>
                            recursive(
                                id         = "id",
                                transform  = lag_roll_transformer_grouped,
                                train_tail = panel_tail(train_tbl, id, FORECAST_HORIZON)
                            )

wflw_fit_tbats_basic <- workflow() |>
                        add_model(model_spec_tbats) |>
                        add_recipe(recipe_spec_date) |>
                        fit(train_tbl)


# stlm_ets model

model_spec_stlm_ets <-  seasonal_reg() |>
                        set_engine("stlm_ets")

model_spec_stlm_ets

wflw_fit_stlm_ets <-  workflow() |>
                      add_model(model_spec_stlm_ets) |>
                      add_recipe(recipe_spec_date) |>
                      fit(train_tbl)

wflw_fit_stlm_ets


wflw_fit_stlm_ets_recursive <-  workflow() |>
                                add_model(model_spec_stlm_ets) |>
                                add_recipe(recipe_spec_date) |>
                                fit(train_tbl) |> recursive(
                                    id         = "id",
                                    transform  = lag_roll_transformer_grouped,
                                    train_tail = panel_tail(train_tbl, id, FORECAST_HORIZON)
                                )

wflw_fit_stlm_ets_recursive


model_spec_arima <-  arima_reg() |>
                    set_engine("auto_arima")
model_spec_arima

wflw_fit_arima <-  workflow() |>
                   add_model(model_spec_arima) |>
                   add_recipe(recipe_spec_date) |>
                   fit(train_tbl)

wflw_fit_arima_recursive <-  workflow() |>
                             add_model(model_spec_arima) |>
                             add_recipe(recipe_spec_date) |>
                             fit(train_tbl) |>
                             recursive(
                                id         = "id",
                                transform  = lag_roll_transformer_grouped,
                                train_tail = panel_tail(train_tbl, id, FORECAST_HORIZON)
                             )


wflw_fit_arima_recursive

# nnetar

model_spec_nnetar <- nnetar_reg()  |>
                     set_engine("nnetar")

model_spec_nnetar

wflw_fit_nnetar <- workflow() |>
                   add_model(model_spec_nnetar) |>
                   add_recipe(recipe_spec_date) |>
                   fit(train_tbl)

wflw_fit_nnetar_recursive <- workflow() |>
                             add_model(model_spec_nnetar) |>
                             add_recipe(recipe_spec_date) |>
                             fit(train_tbl) |>
                             recursive(
                                id         = "id",
                                transform  = lag_roll_transformer_grouped,
                                train_tail = panel_tail(train_tbl, id, FORECAST_HORIZON)
                             )



wflw_fit_nnetar_recursive

# deep learning models

set.seed(123)
model_spec_deepar_gluonts <- deep_ar(
                                id                = "id",
                                freq              = "M",
                                prediction_length = 6,
                                lookback_length   = 6*3,
                                epochs            = 10
                                    ) |>
                                set_engine("gluonts_deepar")

model_spec_deepar_gluonts

set.seed(123)
wflw_fit_deepar_gluonts_recursive <- workflow() |>
                                     add_model(model_spec_deepar_gluonts) |>
                                     add_recipe(recipe_spec_1) |>
                                     fit(train_tbl) |>
                                     recursive(
                                          id         = "id",
                                          transform  = lag_roll_transformer_grouped,
                                          train_tail = panel_tail(train_tbl, id, FORECAST_HORIZON)
                                      )

wflw_fit_deepar_gluonts_recursive

set.seed(123)
wflw_fit_deepar_gluonts <- workflow() |>
                           add_model(model_spec_deepar_gluonts) |>
                           add_recipe(recipe_spec_1) |>
                           fit(train_tbl)


wflw_fit_deepar_gluonts

# * Calibration Results
set.seed(123)
model_spec_gp_forecaster  <- gp_forecaster(
                                id                = "id",
                                freq              = "M",
                                prediction_length = 6,
                                # lookback_length   = 6*3,
                                epochs            = 30
                                        ) |>
                                set_engine("gluonts_gp_forecaster")

model_spec_gp_forecaster

set.seed(123)
wflw_fit_gp_forecaster_recursive  <-  workflow() |>
                                      add_model(model_spec_gp_forecaster) |>
                                      add_recipe(recipe_spec_1) |>
                                      fit(train_tbl) |>
                                      recursive(
                                          id         = "id",
                                          transform  = lag_roll_transformer_grouped,
                                          train_tail = panel_tail(train_tbl, id, FORECAST_HORIZON)
                                      )

wflw_fit_gp_forecaster_recursive


set.seed(123)
wflw_fit_gp_forecaster <- workflow() |>
                          add_model(model_spec_gp_forecaster) |>
                          add_recipe(recipe_spec_1) |>
                          fit(train_tbl)
wflw_fit_gp_forecaster

# * Deep State

set.seed(123)
model_spec_deep_state <- deep_state(
                            id                = "id",
                            freq              = "M",
                            prediction_length = 6,
                            lookback_length   = 6*3,
                            epochs            = 20
                                    ) |>
                            set_engine("gluonts_deepstate")

model_spec_deep_state

set.seed(123)
wflw_fit_deep_state_recursive <- workflow() |>
                                 add_model(model_spec_deep_state) |>
                                 add_recipe(recipe_spec_1) |>
                                 fit(train_tbl) |>
                                 recursive(
                                     id         = "id",
                                     transform  = lag_roll_transformer_grouped,
                                     train_tail = panel_tail(train_tbl, id, FORECAST_HORIZON)
                                 )

wflw_fit_deep_state_recursive

set.seed(123)
wflw_fit_deep_state <- workflow() |>
                       add_model(model_spec_deep_state) |>
                       add_recipe(recipe_spec_1) |>
                       fit(train_tbl)

wflw_fit_deep_state

# Boosted models

model_spec_prophet_boost <- prophet_boost(
                            # Prophet Params
                            # changepoint_num    = 25,
                            # changepoint_range  = 0.8,

                            #   seasonality_weekly = FALSE,
                            #   seasonality_yearly = FALSE,

                            # Xgboost
                            #  mtry           = 0.75,
                            #   min_n          = 20,
                            #   tree_depth     = 3,
                            # learn_rate     = 0.2,
                            #  loss_reduction = 0.15,
                            #  trees          = 300
                                ) |>
                            set_engine("prophet_xgboost")

model_spec_prophet_boost

# Workflow

set.seed(123)
wflw_fit_prophet_boost <-   workflow() |>
                            add_model(model_spec_prophet_boost) |>
                            add_recipe(recipe_spec_1) |>
                            fit(train_tbl)

wflw_fit_prophet_boost

# Recursive
set.seed(123)
wflw_fit_prophet_boost_recursive <- workflow() |>
                                    add_model(model_spec_prophet_boost) |>
                                    add_recipe(recipe_spec_1) |>
                                    fit(train_tbl) |> recursive(
                                        id         = "id",
                                        transform  = lag_roll_transformer_grouped,
                                        train_tail = panel_tail(train_tbl, id, FORECAST_HORIZON)
                                    )
wflw_fit_prophet_boost_recursive

model_spec_arima_boost <- arima_boost(
                            # seasonal_period = 1,
                            #  mtry           = 0.75,
                            # min_n          = 20,
                            # tree_depth     = 3,
                            # learn_rate     = 0.25,
                            # loss_reduction = 0.15,
                            # trees          = 300
                                    ) |>
                            set_engine("auto_arima_xgboost")

set.seed(123)
wflw_fit_arima_boost <- workflow() |>
                        add_model(model_spec_arima_boost) |>
                        add_recipe(recipe_spec_1) |>
                        fit(train_tbl)


wflw_fit_arima_boost


# Recursive

wflw_fit_arima_boost_recursive <- workflow() |>
                                  add_model(model_spec_arima_boost) |>
                                  add_recipe(recipe_spec_1) |>
                                  fit(train_tbl) |>
                                  recursive(
                                      id         = "id",
                                      transform  = lag_roll_transformer_grouped,
                                      train_tail = panel_tail(train_tbl, id, FORECAST_HORIZON)
                                  )

wflw_fit_arima_boost_recursive

## Workflows ----

## XGBoost Model ----

model_spec_xgboost <- boost_tree(
                                mode       = "regression",
                                learn_rate = 0.75,
                                min_n      = 1,
                                tree_depth = 12,
                                loss_reduction = 0.001
                            ) |>
                                set_engine("xgboost")


wflw_fit_xgboost_lag <- workflow() |>
                        add_model(model_spec_xgboost) |>
                        add_recipe(recipe_spec_lag) |>
                        fit(train_tbl) |>
                        recursive(
                            id         = "id",
                            transform  = lag_roll_transformer_grouped,
                            train_tail = panel_tail(train_tbl, id, FORECAST_HORIZON)
                        )

wflw_fit_xgboost_calendar <- workflow() |>
                             add_model(model_spec_xgboost) |>
                             add_recipe(recipe_spec_calendar) |>
                             fit(train_tbl)

wflw_fit_xgboost_hybrid <-  workflow() |>
                            add_model(model_spec_xgboost) |>
                            add_recipe(recipe_spec_hybrid) |>
                            fit(train_tbl) |>
                            recursive(
                                id = "id",
                                transform = lag_roll_transformer_grouped,
                                train_tail = panel_tail(train_tbl, id, FORECAST_HORIZON)
                            )


## Lightgbm ----

model_spec_lighgbm <- boost_tree(mode = "regression") |>
                      set_engine("lightgbm")

wflw_fit_lightgbm_lag <- workflow() |>
                         add_model(model_spec_lighgbm) |>
                         add_recipe(recipe_spec_lag) |>
                         fit(train_tbl) |>
                         recursive(
                            id         = "id",
                            transform  = lag_roll_transformer_grouped,
                            train_tail = panel_tail(train_tbl, id, FORECAST_HORIZON)
                        )

wflw_fit_lightgbm_calendar <- workflow() |>
                              add_model(model_spec_lighgbm) |>
                              add_recipe(recipe_spec_calendar) |>
                              fit(train_tbl)

wflw_fit_lightgbm_hybrid <- workflow() |>
                            add_model(model_spec_lighgbm) |>
                            add_recipe(recipe_spec_hybrid) |>
                            fit(train_tbl) |>
                            recursive(
                                id = "id",
                                transform = lag_roll_transformer_grouped,
                                train_tail = panel_tail(train_tbl, id, FORECAST_HORIZON)
                            )
# Random Forest

model_spec_rf <- rand_forest(
                            mode = "regression",
                            #mtry = 25,
                            #trees = 1000,
                            # min_n = 25
                                )|>
                            set_engine("randomForest")


wflw_fit_rf_lag <- workflow() |>
    add_model(model_spec_rf) |>
    add_recipe(recipe_spec_lag) |>
    fit(train_tbl) |>
    recursive(
        id         = "id",
        transform  = lag_roll_transformer_grouped,
        train_tail = panel_tail(train_tbl, id, FORECAST_HORIZON)
    )

wflw_fit_rf_calendar <- workflow() |>
    add_model(model_spec_rf) |>
    add_recipe(recipe_spec_calendar) |>
    fit(train_tbl)

wflw_fit_rf_hybrid <- workflow() |>
    add_model(model_spec_rf) |>
    add_recipe(recipe_spec_hybrid) |>
    fit(train_tbl) |>
    recursive(
        id = "id",
        transform = lag_roll_transformer_grouped,
        train_tail = panel_tail(train_tbl, id, FORECAST_HORIZON)
    )


# C. SVM

model_spec_svm <- svm_rbf(
                        mode = "regression",
                        margin = 0.001
                    ) |>
                        set_engine("kernlab")

# * Workflows

wflw_fit_svm_lag <- workflow() |>
                    add_model(model_spec_svm) |>
                    add_recipe(recipe_spec_lag) |>
                    fit(train_tbl) |>
                    recursive(
                        id         = "id",
                        transform  = lag_roll_transformer_grouped,
                        train_tail = panel_tail(train_tbl, id, FORECAST_HORIZON)
                    )

wflw_fit_svm_calendar <- workflow() |>
    add_model(model_spec_svm) |>
    add_recipe(recipe_spec_calendar) |>
    fit(train_tbl)

wflw_fit_svm_hybrid <- workflow() |>
    add_model(model_spec_svm) |>
    add_recipe(recipe_spec_hybrid) |>
    fit(train_tbl) |>
    recursive(
        id = "id",
        transform = lag_roll_transformer_grouped,
        train_tail = panel_tail(train_tbl, id, FORECAST_HORIZON)
    )

## Calibration Results----

calibration_tbl <- modeltime_table(
                                wflw_fit_ets_basic,# 1
                                wflw_fit_ets_recursive, #2
                                wflw_fit_theta_recursive, #3
                                wflw_fit_theta_basic, #4
                                wflw_fit_glmnet_lag, #5
                                wflw_fit_glmnet_calendar, #6
                                wflw_fit_glmnet_hybrid, #7
                                wflw_fit_tbats_recursive, #8
                                wflw_fit_tbats_basic, #9
                                wflw_fit_stlm_ets, #10
                                wflw_fit_stlm_ets_recursive, #11
                                wflw_fit_arima,  #12
                                wflw_fit_arima_recursive, #13
                                wflw_fit_nnetar,  #14
                                wflw_fit_nnetar_recursive, #15
                                wflw_fit_deepar_gluonts, #16
                                wflw_fit_deepar_gluonts_recursive, #17
                                wflw_fit_gp_forecaster,  #18
                                wflw_fit_gp_forecaster_recursive, #19
                                wflw_fit_deep_state,  #20
                                wflw_fit_deep_state_recursive, #21
                                wflw_fit_prophet_boost,  #22
                                wflw_fit_prophet_boost_recursive, #23
                                wflw_fit_xgboost_lag,  #24
                                wflw_fit_xgboost_calendar, #25
                                wflw_fit_xgboost_hybrid, # 26
                                wflw_fit_lightgbm_lag, #27
                                wflw_fit_lightgbm_calendar, #28
                                wflw_fit_lightgbm_hybrid, #29
                                wflw_fit_rf_lag, #30
                                wflw_fit_rf_calendar, #31
                                wflw_fit_rf_hybrid, #32
                                wflw_fit_svm_lag, #33
                                wflw_fit_svm_calendar, #34
                                wflw_fit_svm_hybrid, #35
                                wflw_fit_arima_boost, #36
                                wflw_fit_arima_boost_recursive #37
                                )  |>
                        modeltime_calibrate(test_tbl, id = "id")


 total_calibration_tbl  <- combine_modeltime_tables(
                                calibration_tbl,
                                calibration_baseline_models_tbl) |>
                            modeltime_calibrate(test_tbl, id = "id")

 total_calibration_tbl

# Global Accuracy

 total_calibration_tbl |>
    modeltime_accuracy(acc_by_id = FALSE) |>  arrange(rmse) |>
    table_modeltime_accuracy()

 total_calibration_tbl  |>
    modeltime_accuracy(acc_by_id = TRUE) |>  group_by(id) |>  arrange(rmse) |>
    table_modeltime_accuracy()


# Global Confidence Intervals

total_calibration_tbl  |> filter(.model_id %in% c(16,20,52,30,22,28)) |>
            modeltime_forecast(
                        new_data    = test_tbl,
                        actual_data = data_prepared_tbl,
                        keep_data   = TRUE) |>
                        group_by(id) |>
                    # filter_by_time(.start_date = "2019") |>
                    group_by(id) |>
                    plot_modeltime_forecast(
                        .facet_ncol    = 2,
                        .trelliscope   = TRUE
                    )


## Cross validation global model-----

# to by tune
# 15 recursive Deep AR,
# 31  ARIMA wit XGBOOST ERRORS,
# 33  Recursive Glimnet

set.seed(123)
resamples_kfold <- vfold_cv(train_tbl,v = 10)

resamples_kfold |>
    tk_time_series_cv_plan() |>
    plot_time_series_cv_plan(date, value, .facet_ncol = 3)

? boost_tree

model_spec_lightgbm_tune <- boost_tree(
                                        mode       = "regression",
                                        learn_rate = tune(),
                                        tree_depth = tune(),
                                       trees = tune()

                                    ) |>
                            set_engine("lightgbm")


model_spec_lightgbm_tune

library(doFuture)
registerDoFuture()
n_cores <- parallel::detectCores()
n_cores

plan(
    strategy = cluster,
    workers  = parallel::makeCluster(n_cores)
)

## K-Fold Cross Validation ----

# tune lightgbm

wflw_fit_lightgbm <- total_calibration_tbl |>
                               pluck_modeltime_model(28)

wflw_fit_lightgbm


set.seed(123)
tune_results_wflw_fit_lightgbm_kfold <-  wflw_fit_lightgbm   |>
                                         update_model(model_spec_lightgbm_tune) |>
                                         tune_grid(
                                            resamples = resamples_kfold,
                                            #grid      = grid,
                                            metrics   = default_forecast_accuracy_metric_set(),
                                            control   = control_grid(verbose = FALSE, save_pred = TRUE)
                                         )

tune_results_wflw_fit_lightgbm_kfold


g_1 <- tune_results_wflw_fit_lightgbm_kfold |>
       autoplot() +
       geom_smooth(se = FALSE)

ggplotly(g_1)

set.seed(123)
wflw_fit_lightgbm_calendar_tune <- wflw_fit_lightgbm  |>
                                          update_model(model_spec_lightgbm_tune) |>
                                          finalize_workflow(
                                              tune_results_wflw_fit_lightgbm_kfold |>
                                                    show_best(metric = "rmse") |>
                                                    dplyr::slice(1)
                                                    ) |> fit(train_tbl)

wflw_fit_lightgbm_calendar_tune


# tune random forest

plan(
    strategy = cluster,
    workers  = parallel::makeCluster(n_cores)
)

wflw_fit_recursive_rf <- calibration_tbl |> pluck_modeltime_model(30)
wflw_fit_recursive_rf

model_spec_rf_tune <- rand_forest(
                        mode = "regression",
                        mtry = tune(),
                        trees = tune(),
                        min_n = tune()
                    )|>
                    set_engine("randomForest")

model_spec_rf_tune


start_time <- Sys.time()

set.seed(123)
tune_results_rf_kfold <- wflw_fit_recursive_rf |>
                         update_model(model_spec_rf_tune) |>
                         tune_grid(
                            resamples = resamples_kfold,
                            metrics   = default_forecast_accuracy_metric_set(),
                            control   = control_grid(verbose = TRUE, save_pred = TRUE)
                         )

tune_results_rf_kfold

Sys.time() - start_time
parallel_stop()

set.seed(123)
wflw_fit_recursive_rf_tune <-  wflw_fit_lightgbm  |>
                                    update_model(model_spec_rf_tune) |>
                                    finalize_workflow(
                                        tune_results_rf_kfold |>
                                            show_best(metric = "rmse") |>
                                            dplyr::slice(1)) |> fit(train_tbl)

wflw_fit_recursive_rf_tune


## Time Series Cross Validation----

wflw_fit_nnetar_recur   <- total_calibration_tbl |>
                           pluck_modeltime_model(15)

wflw_fit_nnetar_recur



? time_series_cv

train_tbl |> count(date)

resamples_tscv_lag <- time_series_cv(
                        data = train_tbl |> drop_na(),
                        date_var    = date,
                        cumulative  = TRUE,
                        initial     = "24 months",
                        assess      = "24 months",
                        skip        = "2 months",
                        slice_limit = 10
                    )

resamples_tscv_lag |>
    tk_time_series_cv_plan() |>
    plot_time_series_cv_plan(date, value)


model_spec_nnetar <- nnetar_reg(
                        seasonal_period = 7,
                        non_seasonal_ar = tune(id = "non_seasonal_ar"),
                        seasonal_ar     = tune(),
                        hidden_units    = tune(),
                        num_networks    = 10,
                        penalty         = tune(),
                        epochs          = 50
                         ) |>
                     set_engine("nnetar")


# Round 1

?grid_latin_hypercube()

set.seed(123)
grid_spec_nnetar_1 <- grid_latin_hypercube(
                        parameters(model_spec_nnetar),
                        size = 15)
grid_spec_nnetar_1

# * Tune
# - Expensive Operation
# - Parallel Processing is essential

?tune_grid

# Workflow - Tuning

wflw_tune_nnetar <- wflw_fit_nnetar |>
                    update_recipe(recipe_spec_date) |>
                    update_model(model_spec_nnetar)

wflw_tune_nnetar


library(doFuture)
registerDoFuture()
n_cores <- parallel::detectCores()
n_cores

plan(
    strategy = cluster,
    workers  = parallel::makeCluster(n_cores)
)



# Round 1
plan(strategy = sequential)
start_time <- Sys.time()
set.seed(123)
tune_results_nnetar_1 <- wflw_tune_nnetar |>
    tune_grid(
        resamples = resamples_tscv_lag,
        grid      = grid_spec_nnetar_1,
        metrics   = default_forecast_accuracy_metric_set(),
        control   = control_grid(verbose = TRUE, save_pred = TRUE)
    )
Sys.time() - start_time



## Retrain & Assess

set.seed(123)
wflw_fit_nnetar_tscv_tune <- wflw_tune_nnetar |>
                             finalize_workflow(
                                tune_results_nnetar_1 |>
                                    show_best(metric = "rmse", n = Inf) |>
                                    dplyr::slice(1)
                             ) |>
                             fit(train_tbl)

wflw_fit_nnetar_tscv_tune

calibration_tune_models <- modeltime_table(
                            wflw_fit_lightgbm_calendar_tune,
                            wflw_fit_recursive_rf_tune,
                            wflw_fit_nnetar_tscv_tune
                            ) |>
                            modeltime_calibrate(test_tbl, id = "id") |>
                            update_model_description(1, "Lightgbm-Tuned") |>
                            update_model_description(2, "RF_recursive-Tuned")    |>
                            update_model_description(3, "NNAR-Tuned")

calibration_tune_models

# * Global Accuracy

combine_modeltime_tables(
                        calibration_tune_models,
                        total_calibration_tbl
                        ) |>
                        modeltime_calibrate(test_tbl, id = "id")|>
                        modeltime_accuracy(acc_by_id = FALSE) |>  arrange(rmse) |>
                        table_modeltime_accuracy()


# Round 2 if needed

# Visualize Results

g <- tune_results_nnetar_1 |>
    autoplot() +
    geom_smooth(se = FALSE)

ggplotly(g)

# Round 2

set.seed(123)
grid_spec_nnetar_2 <- grid_latin_hypercube(
                         hidden_units(range = c(6, 9)),
                         penalty(range = c(-5, -9), trans = scales::log10_trans()),
                         non_seasonal_ar(range = c(3, 5)),
                         seasonal_ar(range = c(1, 2)),
                         size = 15
                            )
grid_spec_nnetar_2


start_time <- Sys.time()
set.seed(123)
tune_results_nnetar_2 <- wflw_tune_nnetar |>
                            tune_grid(
                                resamples = resamples_tscv_lag,
                                grid      = grid_spec_nnetar_2,
                                metrics   = default_forecast_accuracy_metric_set(),
                                control   = control_grid(verbose = TRUE, save_pred = TRUE)
                                )
Sys.time() - start_time

tune_results_nnetar_2

# * Reset Sequential Plan

plan(strategy = sequential)

set.seed(123)
wflw_fit_nnetar_tscv_tune_2d <- wflw_tune_nnetar |>
                                finalize_workflow(
                                tune_results_nnetar_1 |>
                                    show_best(metric = "rmse", n = Inf) |>
                                    dplyr::slice(1)
                                    ) |>
                                fit(train_tbl)

wflw_fit_nnetar_tscv_tune_2d


calibration_tune_models <- modeltime_table(
                                wflw_fit_lightgbm_calendar_tune,
                                wflw_fit_recursive_rf_tune,
                                wflw_fit_nnetar_tscv_tune ,
                                wflw_fit_nnetar_tscv_tune_2d
                                ) |>
                            modeltime_calibrate(test_tbl, id = "id") |>
                            update_model_description(1, "Lightgbm-Tuned") |>
                            update_model_description(2, "RF_recursive-Tuned")    |>
                            update_model_description(3, "NNAR-Tuned") |>
                            update_model_description(4, "NNAR-Tuned_V2")

calibration_tune_models |> modeltime_accuracy(acc_by_id = FALSE) |>  arrange(rmse) |>
    table_modeltime_accuracy()

# Save all models and tabels

prepared_tbl_artifacts <-  list (
                            # Data
                             data = list(
                                data_prepared_tbl    =  data_prepared_tbl,
                                data_future_tbl = data_future_tbl,
                                train_tbl = train_tbl,
                                test_tbl = test_tbl,
                                data_extended_tbl = data_extended_tbl
                                ),
                            # Recipes
                            recipes = list(
                                recipe_spec_date = recipe_spec_date,
                                recipe_spec_1     = recipe_spec_1 ,
                                recipe_spec_lag    = recipe_spec_lag,
                                recipe_spec_calendar = recipe_spec_calendar,
                                recipe_spec_hybrid = recipe_spec_hybrid
                            )
                            )

 write_rds(prepared_tbl_artifacts,"models/prepared_tbl_artifacts.rds")

#dump(c("calibrate_and_plot"), file = "scripts/calibrate_and_plot.R")

#source("scripts/0calibrate_and_plot.R")

# * Global Accuracy

all_models_combine_accuracy_tbl <- combine_modeltime_tables(
                                     calibration_tune_models,
                                     total_calibration_tbl
                                        ) |>
                                     modeltime_calibrate(test_tbl, id = "id")|>
                                     modeltime_accuracy(acc_by_id = FALSE) |>  arrange(rmse) |>
                                     table_modeltime_accuracy()

all_models_combine_accuracy_tbl

## Global Ensemble models-----

submodels_tbl <-  combine_modeltime_tables(
                    calibration_tune_models,
                    total_calibration_tbl
                        ) |>
                    modeltime_calibrate(test_tbl, id = "id") |> filter(.model_id %in% c(24,20,56,2))

submodels_tbl |>  modeltime_accuracy(acc_by_id = FALSE) |>  arrange(rmse)
submodels_tbl |>  modeltime_accuracy(acc_by_id = TRUE) |>  group_by(id) |> arrange(rmse)


# plot the best models

submodels_tbl|> modeltime_forecast(
    new_data    = test_tbl,
    actual_data = data_prepared_tbl,
    keep_data   = TRUE) |>
    group_by(id) |>
    plot_modeltime_forecast(
        .facet_ncol    = 2,
        .trelliscope   = TRUE)

# evaluates stability of  models

? time_series_cv

train_tbl |> count(date)

resamples_tscv <- time_series_cv(
    data = train_tbl |> drop_na(),
    date_var    = date,
    cumulative  = TRUE,
    initial     = "24 months",
    assess      = "24 months",
    skip        = "2 months",
    slice_limit = 10
)

resamples_tscv |>
    tk_time_series_cv_plan() |>
    plot_time_series_cv_plan(date, value)

# model 1
start_time <- Sys.time()
resample_models_1 <- submodels_tbl|> filter(.model_id==2) |>
    modeltime_fit_resamples(
        resamples = resamples_tscv,
        control   = control_resamples(save_pred = TRUE, allow_par = TRUE)
    )

resample_models_1


start_time <- Sys.time()
resample_models_2 <- submodels_tbl|> filter(.model_id==20) |>
    modeltime_fit_resamples(
        resamples = resamples_tscv,
        control   = control_resamples(save_pred = TRUE, allow_par = TRUE)
    )


resample_models_2

start_time <- Sys.time()
resample_models_3 <- submodels_tbl|> filter(.model_id==24) |>
    modeltime_fit_resamples(
        resamples = resamples_tscv,
        control   = control_resamples(save_pred = TRUE, allow_par = TRUE)
    )


resample_models_3


start_time <- Sys.time()
resample_models_4 <- submodels_tbl|> filter(.model_id==56) |>
    modeltime_fit_resamples(
        resamples = resamples_tscv,
        control   = control_resamples(save_pred = TRUE, allow_par = TRUE)
    )

resample_models_4


resample_models_1_rs <- resample_models_1 |> modeltime_resample_accuracy(
    metric_set = metric_set(mae, rmse, rsq),
    summary_fns= list(mean=mean,sd=sd)
        )

resample_models_2_rs <-resample_models_2 |> modeltime_resample_accuracy(
    metric_set = metric_set(mae, rmse, rsq),
    summary_fns= list(mean=mean,sd=sd)
)


resample_models_3_rs <-resample_models_3 |> modeltime_resample_accuracy(
    metric_set = metric_set(mae, rmse, rsq),
    summary_fns= list(mean=mean,sd=sd)
)

resample_models_4_rs <-resample_models_4 |> modeltime_resample_accuracy(
    metric_set = metric_set(mae, rmse, rsq),
    summary_fns= list(mean=mean,sd=sd)
)

ensemble_stability_metrics <- bind_rows(resample_models_1_rs,resample_models_2_rs,resample_models_3_rs,resample_models_4_rs)
ensemble_stability_metrics


ensemble_weighted <- submodels_tbl |> # filter(.model_id %in% c(2,24)) |>
                                filter(.model_id %in% c(2,56)) |>
                     ensemble_weighted(
                         loadings = c(8,2)
                         ) |>
                     modeltime_calibrate(test_tbl, id = "id")

ensemble_weighted |>  modeltime_accuracy(acc_by_id = FALSE) |>  arrange(rmse) |>
    table_modeltime_accuracy()


# drop DL models as there are some problem with short prediction

 submodels_tbl <- submodels_tbl |> filter(.model_id %in% c(2,56))

combine_modeltime_tables(
             submodels_tbl,
             ensemble_weighted ) |>
        modeltime_calibrate(test_tbl, id = "id") |>
      modeltime_accuracy(acc_by_id = FALSE) |>  arrange(rmse) |>
        table_modeltime_accuracy()


ensemble_average <- submodels_tbl |>
                     ensemble_average(
                         type = c("mean")
                     ) |>
                    modeltime_calibrate(test_tbl, id = "id")

ensemble_average |>  modeltime_accuracy(acc_by_id = FALSE) |>  arrange(rmse) |>
                     table_modeltime_accuracy()


ensemble_median <- submodels_tbl |>
                     ensemble_average(
                                     type = c("median")
                                 ) |>
                                 modeltime_calibrate(test_tbl, id = "id")

ensemble_median |>  modeltime_accuracy(acc_by_id = FALSE) |>  arrange(rmse) |>
     table_modeltime_accuracy()


# check accuracy of ensemble models

ensemble_mix_accuracy_tbl <- combine_modeltime_tables(
                     submodels_tbl,
                     ensemble_weighted,
                     ensemble_average,
                     ensemble_median
                     ) |>
                     modeltime_calibrate(test_tbl, id = "id") |>
                     modeltime_accuracy(acc_by_id = FALSE) |>  arrange(rmse)

ensemble_mix_accuracy_tbl

ensemble_mix <- combine_modeltime_tables(
                        submodels_tbl,
                        ensemble_weighted,
                        ensemble_average,
                        ensemble_median
                                        ) |> modeltime_calibrate(test_tbl, id = "id")


ensemble_mix |> modeltime_forecast(
        new_data    = test_tbl,
        actual_data = data_prepared_tbl,
        keep_data   = TRUE) |>
        #select(date,id,.value) |>
        group_by(id) |>
        plot_modeltime_forecast(
            .facet_ncol    = 2,
            .trelliscope   = TRUE)

## global model forecast----

the_best_global_modele_on_test <- ensemble_mix |> filter(.model_id == 5) |> modeltime_forecast(
                                    new_data    = test_tbl,
                                    actual_data = data_prepared_tbl,
                                    keep_data   = TRUE) |>
                                    #select(date,id,.value) |>
                                    group_by(id) |>
                                    plot_modeltime_forecast(
                                        .facet_ncol    = 2,
                                        .trelliscope   = TRUE)

the_best_global_modele_on_test


the_best_global_model_global_accuracy <- ensemble_mix |> filter(.model_id == 5) |>
                                         modeltime_calibrate(test_tbl, id = "id") |>
                                         modeltime_accuracy(acc_by_id = FALSE) |>  arrange(rmse)
the_best_global_model_global_accuracy


the_best_global_model_local_accuracy <- ensemble_mix |> filter(.model_id == 5) |>
                                        modeltime_calibrate(test_tbl, id = "id") |>
                                        modeltime_accuracy(acc_by_id = TRUE) |> group_by(id) |>  arrange(rmse)

the_best_global_model_local_accuracy |> ungroup () |> summarise(total_forecast = mean(rmse))


the_best_global_modele_on_test |> write_rds("models/the_best_global_model_on_test.rds")

the_best_global_model_global_accuracy |> write_rds("models/the_best_global_model_global_accuracy.rds")

the_best_global_model_local_accuracy |> write_rds("models/the_best_global_model_local_accuracy.rds")


global_model_the_best_model  <- ensemble_mix |> filter(.model_id == 5)
global_model_the_best_model  |> write_rds("models/global_model_the_best_model.rds")

# ITERATIVE FORECASTING -----

# TRANSFORMER

# *** Important Concept: Transformer Function

# source data_extended_tbl
FORECAST_HORIZON <- as.integer(24)
prepared_tbl_artifacts  <- read_rds("models/prepared_tbl_artifacts.rds")
data_extended_tbl  <- prepared_tbl_artifacts$data$data_prepared_tbl
data_extended_tbl |> glimpse()

transformer_function <- function(data) {
                        data |>
                            tk_augment_lags(
                                value,
                                .lags = 1:FORECAST_HORIZON
                            ) |>
                            tk_augment_slidify(
                                contains("lag12"),
                                .f       = ~mean(.x, na.rm = T),
                                .period  = 12,
                                .partial = TRUE
                            )
                            }
# DATA PREPARED

data_prepared_iterative_tbl <- data_extended_tbl |>
                            group_by(id) |>
                            transformer_function() |>
                            # ***
                            # FIX 1: REMOVE DROP NA
                            # Issue: Causes future data to be removed because of
                            # missing values in lags (which are ok)
                            #drop_na(-value) |>
                                # END FIX
                            ungroup()

data_prepared_iterative_tbl  |> glimpse()

## NESTED TIME SERIES DATA ----

# - How to prepare the data for the "nested workflow"

nested_data_tbl <-  data_prepared_iterative_tbl  |>
                    nest_timeseries(
                        .id_var        = id,
                        .length_future = 24
                    ) |>
                    split_nested_timeseries(
                        .length_test = 24
                    )

nested_data_tbl

extract_nested_train_split(nested_data_tbl) |> glimpse()
extract_nested_test_split(nested_data_tbl) |> glimpse()

# MODELING
#  Recipe

iterative_recipe_spec_date <- recipe(value ~ ., data = extract_nested_train_split(nested_data_tbl) |> select(date,value))
iterative_recipe_spec_date |> prep() |> juice() |> glimpse()

iterative_recipe_spec_1 <- recipe(value ~ ., data = extract_nested_train_split(nested_data_tbl)) |> step_rm(starts_with("lag"))
iterative_recipe_spec_1 |> prep() |> juice() |> glimpse()

iterative_recipe_spec_lag <- recipe(value ~ ., data = extract_nested_train_split(nested_data_tbl)) |>
    step_dummy(all_nominal()) |>
    step_rm(date) |>
    step_zv(all_predictors())

iterative_recipe_spec_lag |> prep() |> juice() |> glimpse()

# ** Calendar + Long-Term Rolling Recipe

iterative_recipe_spec_calendar <- recipe(
    value ~ .,
    data = extract_nested_train_split(nested_data_tbl) |> select(-contains("lag"), contains("roll"))) |>
    step_timeseries_signature(date) |>
    step_dummy(all_nominal(), one_hot = TRUE) |>
    step_normalize(date_index.num, starts_with("date_year")) |>
    step_rm(date) |>
    step_zv(all_predictors())


iterative_recipe_spec_calendar |> prep() |> juice() |> glimpse()

# ** Hybrid Short-Term Lag + Calendar Recipe

iterative_recipe_spec_hybrid <- recipe(value ~ ., data = extract_nested_train_split(nested_data_tbl)) |>
                            step_timeseries_signature(date) |>
                            step_dummy(all_nominal(), one_hot = TRUE) |>
                            step_normalize(date_index.num, starts_with("date_year")) |>
                            step_rm(date) |>
                            step_zv(all_predictors())

iterative_recipe_spec_hybrid |> prep() |> juice() |> glimpse()

## Baseline Algorithms ----
# - Used to make simple window models (mean, median, seasonal naive)


# Window_3

iterative_wflw_window_6  <-   workflow() |>
    add_model(window_reg(window_size = 6) |> set_engine("window_function", window_function = mean)) |>
    add_recipe(recipe(value ~ ., extract_nested_train_split(nested_data_tbl)))

# Window_1

iterative_wflw_window_1  <-   workflow() |>
    add_model(window_reg(window_size = 1) |> set_engine("window_function", window_function = mean)) |>
    add_recipe(recipe(value ~ ., extract_nested_train_split(nested_data_tbl)))


# * Seasonal NAIVE

iterative_wflw_seasonal_naive  <- workflow() |>
    add_model(naive_reg(seasonal_period = 12) |> set_engine("snaive")) |>
    add_recipe(recipe(value ~ ., extract_nested_train_split(nested_data_tbl)))


baseline_nested_modeltime_tbl <- nested_data_tbl |>
                                modeltime_nested_fit(

                                    model_list = list(
                                        iterative_wflw_window_6,
                                        iterative_wflw_window_1,
                                        iterative_wflw_seasonal_naive
                                    ),

                                    control = control_nested_fit(
                                        verbose   = TRUE,
                                        allow_par = FALSE  # <-- SET TO TRUE IF WANT TO RUN ALL 102 IN PARALLEL
                                    )
                                )

baseline_nested_modeltime_tbl

baseline_nested_modeltime_tbl |>
    extract_nested_test_accuracy() |>
    group_by(id) |>
    slice_min(order_by = rmse, n = 1)


##  Iterative hyperparameter tuning ----

## XGBoost Models ----

iterative_wflw_xgb_1 <- workflow() |>
    add_model(boost_tree("regression",
                       #  learn_rate = 0.35
                         ) |> set_engine("xgboost")) |>
    add_recipe(iterative_recipe_spec_calendar)

#iterative_wflw_xgb_2 <- workflow() |>
#    add_model(boost_tree("regression", learn_rate = 0.50) |> set_engine("xgboost")) |>
#    add_recipe(iterative_recipe_spec_calendar)


grid_tbl <- tibble(
                learn_rate = c(0.010, 0.100, 0.310, 0.500, 0.75)
                    ) |>
                create_model_grid(
                    f_model_spec = boost_tree,
                    engine_name  = "xgboost",
                    mode         = "regression"
                )

grid_tbl$.models

library(workflowsets)

wf_set <- workflow_set(
    preproc = list(
        iterative_recipe_spec_calendar,
        iterative_recipe_spec_hybrid,
        iterative_recipe_spec_lag
    ),
    models = grid_tbl$.models,
    cross  = TRUE
)
wf_set

wf_grid <- wf_set$info |>
    bind_rows() |>
    bind_cols(wf_set[,1])
wf_grid

# * Model List

iterative_xgboost_models  <- c(
                                 list(
                                    iterative_wflw_xgb_1#,
                                    # iterative_wflw_xgb_2
                                 ),
                                wf_grid$workflow
                            )



iterative_xgboost_models[[13]]

iterative_xgboost_models|> pluck(-13)

iterative_xgboost_models_tbl  <- nested_data_tbl |>
                                 modeltime_nested_fit(
                                    model_list = iterative_xgboost_models,
                                    control = control_nested_fit(
                                        verbose   = TRUE,
                                        allow_par = TRUE  # <-- SET TO TRUE IF WANT TO RUN ALL 102 IN PARALLEL
                                    ))

# # iterative_xgboost_models_tbl |> extract_nested_modeltime() |> glimpse()



iterative_xgboost_models_tbl |>
    extract_nested_test_forecast() |>
    group_by(id) |>
    plot_modeltime_forecast()

the_best_iterative_xgboost_models_tbl <- iterative_xgboost_models_tbl |>
    extract_nested_test_accuracy()  |> filter(.model_id != 8) |>
    group_by(id) |>
    slice_min(order_by = rmse, n = 1)


the_best_iterative_xgboost_models_tbl |> group_by(.model_id) |>  count(.model_id)


iterative_wflw_xgb_3 <-iterative_xgboost_models |> pluck(3)
iterative_wflw_xgb_3

iterative_wflw_xgb_5 <-iterative_xgboost_models |> pluck(5)
iterative_wflw_xgb_5

iterative_wflw_xgb_6 <-iterative_xgboost_models |> pluck(6)
iterative_wflw_xgb_6

iterative_wflw_xgb_9 <-iterative_xgboost_models |> pluck(9)
iterative_wflw_xgb_9

iterative_wflw_xgb_13 <-iterative_xgboost_models |> pluck(13)
iterative_wflw_xgb_13

iterative_wflw_xgb_15 <-iterative_xgboost_models |> pluck(15)
iterative_wflw_xgb_15

iterative_wflw_xgb_16 <-iterative_xgboost_models |> pluck(16)
iterative_wflw_xgb_16

# add the best xgboost models into the baseline_nested_modeltime_tbl

## Lightgbm

library(bonsai)

iterative_model_spec_lighgbm <- boost_tree(mode = "regression") |>
                                set_engine("lightgbm")

iterative_wflw_fit_lightgbm_lag <- workflow() |>
     add_model(iterative_model_spec_lighgbm) |>
     add_recipe(iterative_recipe_spec_lag)|>
     fit(extract_nested_train_split(nested_data_tbl)) |>
     recursive(
         transform  = transformer_function,
         train_tail = tail(extract_nested_train_split(nested_data_tbl), n = FORECAST_HORIZON),
         chunk_size = 1
     )

iterative_wflw_fit_lightgbm_calendar <- workflow() |>
     add_model(iterative_model_spec_lighgbm) |>
     add_recipe(iterative_recipe_spec_calendar) |>
     fit(extract_nested_train_split(nested_data_tbl))

iterative_wflw_fit_lightgbm_hybrid <- workflow() |>
     add_model(iterative_model_spec_lighgbm) |>
     add_recipe(iterative_recipe_spec_hybrid) |>
     fit(extract_nested_train_split(nested_data_tbl)) |>
     recursive(
         transform  = transformer_function,
         train_tail = tail(extract_nested_train_split(nested_data_tbl), n = FORECAST_HORIZON),
         chunk_size = 1
        )
#SVM

 iterative_model_spec_svm <- svm_rbf(
     mode = "regression",
     margin = 0.001
 ) |>
     set_engine("kernlab")


 iterative_wflw_fit_svm_lag <- workflow() |>
     add_model(iterative_model_spec_svm) |>
     add_recipe(iterative_recipe_spec_lag) |>
     fit(extract_nested_train_split(nested_data_tbl)) |>
     recursive(
         transform  = transformer_function,
         train_tail = tail(extract_nested_train_split(nested_data_tbl), n = FORECAST_HORIZON),
         chunk_size = 1)

 iterative_wflw_fit_svm_calendar <- workflow() |>
     add_model(iterative_model_spec_svm) |>
     add_recipe(iterative_recipe_spec_calendar) |>
     fit(extract_nested_train_split(nested_data_tbl))

 iterative_wflw_fit_svm_hybrid <- workflow() |>
     add_model(iterative_model_spec_svm) |>
     add_recipe(iterative_recipe_spec_hybrid) |>
     fit(extract_nested_train_split(nested_data_tbl)) |>
     recursive(
         transform  = transformer_function,
         train_tail = tail(extract_nested_train_split(nested_data_tbl), n = FORECAST_HORIZON),
         chunk_size = 1
     )


# * Workflows

# * Classical algorithms

iterative_model_spec_ets  <- exp_smoothing() |>
                             set_engine("ets")

iterative_wflw_fit_ets_recursive <-   workflow() |>
    add_model(iterative_model_spec_ets) |>
    add_recipe(iterative_recipe_spec_date) |>
    fit(extract_nested_train_split(nested_data_tbl)) |>
    recursive(
        transform  = transformer_function,
        train_tail = tail(extract_nested_train_split(nested_data_tbl), n = FORECAST_HORIZON),
        chunk_size = 1)

iterative_wflw_fit_ets_basic <-   workflow() |>
    add_model(iterative_model_spec_ets) |>
    add_recipe(iterative_recipe_spec_date) |>
    fit(extract_nested_train_split(nested_data_tbl))


# theta model

iterative_model_spec_theta <-  exp_smoothing() |>
    set_engine("ets")

iterative_wflw_fit_theta_recursive <-   workflow() |>
    add_model(iterative_model_spec_theta) |>
    add_recipe(iterative_recipe_spec_date) |>
    fit(extract_nested_train_split(nested_data_tbl)) |>
    recursive(
        transform  = transformer_function,
        train_tail = tail(extract_nested_train_split(nested_data_tbl), n = FORECAST_HORIZON),
        chunk_size = 1)

iterative_wflw_fit_theta_basic <- workflow() |>
    add_model(iterative_model_spec_theta) |>
    add_recipe(iterative_recipe_spec_date) |>
    fit(extract_nested_train_split(nested_data_tbl))

# tbats model

iterative_model_spec_tbats <-  seasonal_reg(
    seasonal_period_1 = 7,
    seasonal_period_2 = 30,
    seasonal_period_3 = 365
) |>
    set_engine("tbats")

iterative_wflw_fit_tbats_recursive <- workflow() |>
    add_model(iterative_model_spec_tbats) |>
    add_recipe(iterative_recipe_spec_date) |>
    fit(extract_nested_train_split(nested_data_tbl)) |>
    recursive(
        transform  = transformer_function,
        train_tail = tail(extract_nested_train_split(nested_data_tbl), n = FORECAST_HORIZON),
        chunk_size = 1)

iterative_wflw_fit_tbats_basic <- workflow() |>
    add_model(iterative_model_spec_tbats) |>
    add_recipe(iterative_recipe_spec_date) |>
    fit(extract_nested_train_split(nested_data_tbl))


# stlm_ets model

iterative_model_spec_stlm_ets <-  seasonal_reg() |>
    set_engine("stlm_ets")

iterative_wflw_fit_stlm_ets <-  workflow() |>
    add_model(iterative_model_spec_stlm_ets) |>
    add_recipe(iterative_recipe_spec_date) |>
    fit(extract_nested_train_split(nested_data_tbl))

iterative_wflw_fit_stlm_ets


iterative_wflw_fit_stlm_ets_recursive <-  workflow() |>
    add_model(iterative_model_spec_stlm_ets) |>
    add_recipe(iterative_recipe_spec_date) |>
    fit(extract_nested_train_split(nested_data_tbl)) |>
    recursive(
        transform  = transformer_function,
        train_tail = tail(extract_nested_train_split(nested_data_tbl), n = FORECAST_HORIZON),
        chunk_size = 1)

iterative_model_spec_arima <-  arima_reg() |>
                                set_engine("auto_arima")

iterative_wflw_fit_arima <-  workflow() |>
    add_model(iterative_model_spec_arima) |>
    add_recipe(iterative_recipe_spec_date) |>
    fit(extract_nested_train_split(nested_data_tbl))

iterative_wflw_fit_arima_recursive <-  workflow() |>
    add_model(iterative_model_spec_arima) |>
    add_recipe(iterative_recipe_spec_date) |>
    fit(extract_nested_train_split(nested_data_tbl)) |>
    recursive(
        transform  = transformer_function,
        train_tail = tail(extract_nested_train_split(nested_data_tbl), n = FORECAST_HORIZON),
        chunk_size = 1)

# nnetar

iterative_model_spec_nnetar <- nnetar_reg()  |>
    set_engine("nnetar")

iterative_wflw_fit_nnetar <- workflow() |>
    add_model(iterative_model_spec_nnetar) |>
    add_recipe(iterative_recipe_spec_date) |>
    fit(extract_nested_train_split(nested_data_tbl))

iterative_wflw_fit_nnetar_recursive <- workflow() |>
    add_model(iterative_model_spec_nnetar) |>
    add_recipe(iterative_recipe_spec_date) |>
    fit(extract_nested_train_split(nested_data_tbl)) |>
    recursive(
        transform  = transformer_function,
        train_tail = tail(extract_nested_train_split(nested_data_tbl), n = FORECAST_HORIZON),
        chunk_size = 1)


# Boosted models

iterative_model_spec_prophet_boost <- prophet_boost() |>
    set_engine("prophet_xgboost")

# Workflow
# Prophet Boost

set.seed(123)
iterative_wflw_fit_prophet_boost <-   workflow() |>
    add_model(iterative_model_spec_prophet_boost) |>
    add_recipe(iterative_recipe_spec_1) |>
    fit(extract_nested_train_split(nested_data_tbl))


# Recursive
set.seed(123)
wflw_fit_prophet_boost_recursive <- workflow() |>
    add_model(iterative_model_spec_prophet_boost) |>
    add_recipe(iterative_recipe_spec_1) |>
    fit(extract_nested_train_split(nested_data_tbl)) |>
    recursive(
        transform  = transformer_function,
        train_tail = tail(extract_nested_train_split(nested_data_tbl), n = FORECAST_HORIZON),
        chunk_size = 1)

iterative_model_spec_arima_boost <- arima_boost() |>
    set_engine("auto_arima_xgboost")

# Arima boost
set.seed(123)
iterative_wflw_fit_arima_boost <- workflow() |>
    add_model(iterative_model_spec_arima_boost) |>
    add_recipe(iterative_recipe_spec_1) |>
    fit(extract_nested_train_split(nested_data_tbl))


# Recursive

iterative_wflw_fit_arima_boost_recursive <- workflow() |>
    add_model(iterative_model_spec_arima_boost) |>
    add_recipe(iterative_recipe_spec_1) |>
    fit(extract_nested_train_split(nested_data_tbl)) |>
    recursive(
        transform  = transformer_function,
        train_tail = tail(extract_nested_train_split(nested_data_tbl), n = FORECAST_HORIZON),
        chunk_size = 1)


iterative_wflw_fit_arima_boost_recursive


library(doFuture)
registerDoFuture()
n_cores <- parallel::detectCores()
n_cores

plan(
    strategy = cluster,
    workers  = parallel::makeCluster(n_cores)
)

parallel_start(16)

start_time <- Sys.time()
nested_modeltime_tbl <- nested_data_tbl  |>
    modeltime_nested_fit(

        model_list = list(
        iterative_wflw_window_6,  #1
        iterative_wflw_window_1,  #2
        iterative_wflw_seasonal_naive, #3
        iterative_wflw_fit_prophet_boost, #4
        iterative_wflw_fit_arima_boost, #5
        iterative_wflw_fit_arima_boost_recursive, #6
        iterative_wflw_fit_arima, #7
        iterative_wflw_fit_arima_recursive, #8
        iterative_wflw_fit_stlm_ets, #9
        iterative_wflw_fit_stlm_ets_recursive, #10
        iterative_wflw_fit_tbats_basic,   #11
        iterative_wflw_fit_tbats_recursive, #12
        #iterative_wflw_fit_theta_basic, #13
        iterative_wflw_fit_theta_recursive, #14
        iterative_wflw_fit_ets_basic, #15
        iterative_wflw_fit_ets_recursive, #16
        iterative_wflw_fit_nnetar, #17
        iterative_wflw_fit_nnetar_recursive, #18
        iterative_wflw_fit_svm_lag, #19
        iterative_wflw_fit_svm_calendar, #20
        iterative_wflw_fit_svm_hybrid, #21
        iterative_wflw_fit_lightgbm_lag, #22
        iterative_wflw_fit_lightgbm_calendar, #23
        iterative_wflw_fit_lightgbm_hybrid, #24,
        iterative_wflw_xgb_3, #25
        iterative_wflw_xgb_5, #26
        iterative_wflw_xgb_6, #27
        iterative_wflw_xgb_9, #28
        iterative_wflw_xgb_13, #29
        iterative_wflw_xgb_15, #30
        iterative_wflw_xgb_16 #31
        ),
        control = control_nested_fit(
            verbose   = TRUE,
            allow_par = TRUE
        )
    )


Sys.time() - start_time
nested_modeltime_tbl

parallel_stop()

# * Get Best Models

the_best_iterative_modeltime_tbl  <- nested_modeltime_tbl %>%
                modeltime_nested_select_best(
                    metric                = "rmse",
                    minimize              = TRUE,
                    filter_test_forecasts = TRUE
                )


the_best_iterative_modeltime_tbl %>%
     extract_nested_best_model_report()

the_best_iterative_modeltime_tbl %>%
    extract_nested_best_model_report() |>  summarise(iterative_model_total_rmse = mean(rmse))

# modeltime_nested_select_best(metric = "rmse",  minimize  = TRUE, filter_test_forecasts = TRUE)

iterative_model_local_accuracy   <- the_best_iterative_modeltime_tbl %>%
                                    extract_nested_best_model_report()

iterative_model_local_accuracy |> write_rds("models/iterative_model_local_accuracy.rds")

iterative_model_forecast |> write_rds("models/iterative_model_forecast.rds")

# Global forecast vs Iterative forecast ----

loaded_global_model_local_accuracy <-  read_rds("models/the_best_global_model_local_accuracy.rds")
loaded_global_model_local_accuracy

loaded_global_model_global_accuracy   <- read_rds("models/the_best_global_model_global_accuracy.rds")
loaded_global_model_global_accuracy

loaded_global_model_plot  <-  read_rds("models/the_best_global_model_on_test.rds")
loaded_global_model_plot

loaded_global_model <- read_rds("models/global_model_the_best_model.rds")
loaded_global_model

loaded_prepared_tbl_artifacts <-  read_rds("models/prepared_tbl_artifacts.rds")
loaded_data_prepared_tbl <- loaded_prepared_tbl_artifacts$data$data_prepared_tbl
loaded_data_prepared_tbl

data_future_tbl <- loaded_prepared_tbl_artifacts$data$data_future_tbl
data_future_tbl

## Iterative ----

loaded_iterative_model_local_accuracy <-  read_rds("models/iterative_model_local_accuracy.rds")
loaded_iterative_model_local_accuracy

## Compare accuracy----

loaded_iterative_model_local_accuracy |> summarise(rmse=mean(rmse))
loaded_global_model_local_accuracy |> ungroup() |> select(rmse,-id) |> summarise(rmse=mean(rmse))

loaded_global_model_local_accuracy |>
    left_join(loaded_iterative_model_local_accuracy, by = 'id') |>
    select(id, rmse.x, rmse.y) |>
    rename(global_rmse = rmse.x, iterative_rmse = rmse.y) |>
    mutate(winner = case_when(
        global_rmse < iterative_rmse ~ 'global',
        global_rmse == iterative_rmse ~ 'both',
        TRUE ~ 'iterative'
    ))

## Conclusion----
# Iterative model performed better than global model in 6 out of 10 cases

# * Review Any Errors

nested_best_refit_tbl <- the_best_iterative_modeltime_tbl |>
                         modeltime_nested_refit(
                            control = control_refit(
                                verbose   = TRUE,
                                allow_par = FALSE
                            )
                        )

nested_best_refit_tbl

# * Review Any Errors

nested_best_refit_tbl |> extract_nested_error_report()



# Iterative Forecast output

iterative_model_forecast <- nested_best_refit_tbl |>
    extract_nested_future_forecast() |>  filter(.key =='prediction') |>
    summarise(iterative_forecast = sum(.value))

# * Visualize Iterative Forecast

nested_best_refit_tbl |>
    extract_nested_future_forecast() |>
    group_by(id) |>
    plot_modeltime_forecast(
        .trelliscope = TRUE,
        .legend_show = FALSE
    )


# * Visualize Global Forecast
global_refit_tbl |>
    modeltime_forecast(
        new_data    = data_future_tbl,
        actual_data = data_prepared_tbl,
        keep_data   = TRUE,
        conf_by_id  = TRUE) |>
    group_by(id) |>
    plot_modeltime_forecast(
        .facet_ncol    = 2,
        .trelliscope   = TRUE)






