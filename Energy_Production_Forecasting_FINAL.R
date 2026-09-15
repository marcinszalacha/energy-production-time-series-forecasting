# ENERGY PRODUCTION FORECASTING
# Classical Forecasting | Machine Learning | Deep Learning | Ensembles
# Global vs Series-Specific Forecasting
# Author: Marcin Szalacha


# 01 - Execution Settings ------------------------------------------------------

# Keep expensive sections FALSE while debugging.
# Enable them individually once the basic pipeline works.

RUN_DEEP_LEARNING <- TRUE
RUN_TUNING        <- TRUE
RUN_STACKING      <- TRUE
RUN_NESTED_MODELS <- TRUE
RUN_FOUNDATION_MODELS <- TRUE

FORECAST_HORIZON <- 24
SEED <- 123

set.seed(SEED)


# 02 - Libraries ---------------------------------------------------------------

library(tidyverse)
library(tidymodels)

library(timetk)

library(modeltime)
library(modeltime.ensemble)
library(modeltime.resample)

library(workflowsets)
library(bonsai)

library(xgboost)
library(lightgbm)

library(readxl)
library(plotly)
#library(lubridate)

library(glmnet)
library(randomForest)
library(kernlab)

library(doFuture)
library(future)

tidymodels_prefer()


if (RUN_DEEP_LEARNING) {
  
  library(modeltime.gluonts)
  
}


# 03 - Output Folders ----------------------------------------------------------

dir.create(
  "outputs",
  showWarnings = FALSE
)

dir.create(
  "models",
  showWarnings = FALSE
)


# 04 - Import Data -------------------------------------------------------------

raw_data <- read_excel(
  "data/energy_production.xlsx"
) |>
  
  mutate(
    date = as.Date(Date)
  ) |>
  
  filter_by_time(
    .date_var = date,
    .start_date = "2010-01-01"
  ) |>
  
  select(
    -Date
  )


glimpse(raw_data)


# 05 - Initial Data Diagnostics ------------------------------------------------

raw_data |>
  tk_summary_diagnostics()


raw_data |>
  count(
    MSN
  )


raw_data |>
  count(
    Description
  )


raw_data |>
  count(
    Unit
  )


# 06 - Energy Hierarchy --------------------------------------------------------

energy_data <- raw_data |>
  
  mutate(
    
    energy_production_category = case_when(
      
      str_detect(
        Description,
        regex(
          "Coal|Oil|Gas|Nuclear",
          ignore_case = TRUE
        )
      ) ~ "Fossil Fuels Production",
      
      str_detect(
        Description,
        regex(
          "Geo|Hydro|Solar|Wind|Biomass",
          ignore_case = TRUE
        )
      ) ~ "Renewable Energy Production",
      
      TRUE ~ "Other"
    )
    
  ) |>
  
  filter(
    energy_production_category != "Other"
  ) |>
  
  select(
    date,
    Value,
    energy_production_category,
    Description
  )


glimpse(energy_data)


energy_data |>
  distinct(
    Description
  ) |>
  arrange(
    Description
  )


# 07 - Overall Monthly Production ---------------------------------------------

monthly_all_data <- energy_data |>
  
  pad_by_time(
    date,
    .by = "month"
  ) |>
  
  summarise_by_time(
    .date_var = date,
    .by = "month",
    value = sum(Value)
  )


monthly_all_data |>
  
  plot_time_series(
    .date_var = date,
    .value = value,
    .interactive = FALSE,
    .smooth = FALSE
  )


# 08 - Level 1 Monthly Categories ---------------------------------------------

monthly_data_level_1 <- energy_data |>
  
  group_by(
    energy_production_category
  ) |>
  
  pad_by_time(
    date,
    .by = "month",
    .pad_value = 0
  ) |>
  
  summarise_by_time(
    .date_var = date,
    .by = "month",
    value = sum(Value)
  ) |>
  
  ungroup()


monthly_data_level_1 <- monthly_data_level_1 |>
  
  group_by(
    energy_production_category
  ) |>
  
  mutate(
    
    value_na = if_else(
      value == 0,
      NA_real_,
      value
    ),
    
    value = ts_impute_vec(
      value_na
    )
    
  ) |>
  
  select(
    -value_na
  ) |>
  
  ungroup()


# 09 - Level 2 Monthly Energy Series ------------------------------------------

monthly_data_level_2 <- energy_data |>
  
  group_by(
    Description
  ) |>
  
  pad_by_time(
    date,
    .by = "month",
    .pad_value = 0
  ) |>
  
  summarise_by_time(
    .date_var = date,
    .by = "month",
    value = sum(Value)
  ) |>
  
  ungroup()


monthly_data_level_2 <- monthly_data_level_2 |>
  
  group_by(
    Description
  ) |>
  
  mutate(
    
    value_na = if_else(
      value == 0,
      NA_real_,
      value
    ),
    
    value = ts_impute_vec(
      value_na
    )
    
  ) |>
  
  select(
    -value_na
  ) |>
  
  ungroup()


glimpse(monthly_data_level_2)


# 10 - Historical Time-Series Overview ----------------------------------------

monthly_data_level_2 |>
  
  group_by(
    Description
  ) |>
  
  plot_time_series(
    .date_var = date,
    .value = value,
    .facet_ncol = 2,
    .trelliscope = FALSE,
    .interactive = FALSE,
    .smooth = FALSE
  )


# 11 - Fossil vs Renewable Trend ----------------------------------------------

monthly_data_level_1 |>
  
  group_by(
    energy_production_category
  ) |>
  
  plot_time_series(
    .date_var = date,
    .value = value,
    .facet_ncol = 1,
    .trelliscope = FALSE,
    .interactive = FALSE,
    .smooth = TRUE
  )


# 12 - STL Diagnostics ---------------------------------------------------------

monthly_data_level_2 |>
  
  group_by(
    Description
  ) |>
  
  plot_stl_diagnostics(
    .date_var = date,
    .value = value
  )


# 13 - Seasonal Diagnostics ----------------------------------------------------

major_series <- c(
  "Coal Production",
  "Crude Oil Production",
  "Natural Gas (Dry) Production",
  "Nuclear Electric Power Production",
  "Geothermal Energy Production",
  "Hydroelectric Power Production",
  "Solar Energy Production",
  "Wind Energy Production"
)


monthly_data_level_2 |>
  
  filter(
    Description %in% major_series
  ) |>
  
  group_by(
    Description
  ) |>
  
  plot_seasonal_diagnostics(
    .date_var = date,
    .value = value,
    .title = "Seasonal Diagnostics",
    .feature_set = "auto",
    .geom = "boxplot"
  )


# 14 - ACF Diagnostics ---------------------------------------------------------

monthly_data_level_2 |>
  
  filter(
    Description %in% major_series
  ) |>
  
  group_by(
    Description
  ) |>
  
  plot_acf_diagnostics(
    .date_var = date,
    .value = value,
    .lags = 60
  )


# 15 - Anomaly Diagnostics -----------------------------------------------------

monthly_data_level_2 |>
  
  filter(
    Description %in% major_series
  ) |>
  
  group_by(
    Description
  ) |>
  
  plot_anomaly_diagnostics(
    .date_var = date,
    .value = value
  )


# 16 - Prepare Modelling Dataset ----------------------------------------------

model_data <- monthly_data_level_2 |>
  
  rename(
    id = Description
  )


model_data |>
  count(
    id
  )


# 17 - Extend Into Future ------------------------------------------------------

data_extended_tbl <- model_data |>
  
  group_by(
    id
  ) |>
  
  future_frame(
    date,
    .length_out = FORECAST_HORIZON,
    .bind_data = TRUE
  ) |>
  
  ungroup()


glimpse(data_extended_tbl)


# 18 - Lag Transformer ---------------------------------------------------------

lag_roll_transformer_grouped <- function(data) {
  
  data |>
    
    group_by(
      id
    ) |>
    
    tk_augment_lags(
      value,
      .lags = 1:FORECAST_HORIZON
    ) |>
    
    tk_augment_slidify(
      .value = contains("lag12"),
      .f = ~ mean(.x, na.rm = TRUE),
      .period = 12,
      .partial = TRUE
    ) |>
    
    ungroup()
}


# 19 - Lagged Data -------------------------------------------------------------

data_lagged_tbl <- data_extended_tbl |>
  
  lag_roll_transformer_grouped()


glimpse(data_lagged_tbl)


# 20 - Prepared Historical and Future Data ------------------------------------

data_prepared_tbl <- data_lagged_tbl |>
  
  filter(
    !is.na(value)
  ) |>
  
  drop_na()


data_future_tbl <- data_lagged_tbl |>
  
  filter(
    is.na(value)
  )


data_future_tbl |>
  distinct(
    date
  )


# 21 - Train Test Split --------------------------------------------------------

resamples <- data_prepared_tbl |>
  
  time_series_split(
    date,
    assess = FORECAST_HORIZON,
    cumulative = TRUE
  )


train_tbl <- training(
  resamples
)


test_tbl <- testing(
  resamples
)


resamples |>
  
  tk_time_series_cv_plan() |>
  
  plot_time_series_cv_plan(
    .date_var = date,
    .value = value
  )


# 22 - Training Data Cleaning --------------------------------------------------

train_tbl <- train_tbl |>
  
  group_by(
    id
  ) |>
  
  mutate(
    value = ts_clean_vec(
      value,
      period = 12
    )
  ) |>
  
  ungroup()


# 23 - Model Recipes -----------------------------------------------------------

recipe_spec_date <- recipe(
  
  value ~ .,
  
  data = train_tbl |>
    select(
      date,
      value
    )
)


recipe_spec_no_lag <- recipe(
  
  value ~ .,
  
  data = train_tbl
  
) |>
  
  step_rm(
    starts_with("value_lag")
  )


recipe_spec_lag <- recipe(
  
  value ~ .,
  
  data = train_tbl
  
) |>
  
  step_dummy(
    all_nominal_predictors(),
    one_hot = TRUE
  ) |>
  
  step_rm(
    date
  ) |>
  
  step_zv(
    all_predictors()
  )


recipe_spec_calendar <- recipe(
  
  value ~ .,
  
  data = train_tbl |>
    select(
      -contains("lag")
    )
  
) |>
  
  step_timeseries_signature(
    date
  ) |>
  
  step_rm(
    contains("iso"),
    contains("xts")
  ) |>
  
  step_dummy(
    all_nominal_predictors(),
    one_hot = TRUE
  ) |>
  
  step_normalize(
    date_index.num,
    starts_with("date_year")
  ) |>
  
  step_rm(
    date
  ) |>
  
  step_zv(
    all_predictors()
  )


recipe_spec_hybrid <- recipe(
  
  value ~ .,
  
  data = train_tbl
  
) |>
  
  step_timeseries_signature(
    date
  ) |>
  
  step_rm(
    contains("iso"),
    contains("xts")
  ) |>
  
  step_dummy(
    all_nominal_predictors(),
    one_hot = TRUE
  ) |>
  
  step_normalize(
    date_index.num,
    starts_with("date_year")
  ) |>
  
  step_rm(
    date
  ) |>
  
  step_zv(
    all_predictors()
  )


# 24 - Baseline Models ---------------------------------------------------------

model_mean_fit_6 <- window_reg(
  id = "id",
  window_size = 6
) |>
  
  set_engine(
    "window_function",
    window_function = mean
  ) |>
  
  fit(
    value ~ .,
    data = train_tbl
  )


model_mean_fit_12 <- window_reg(
  id = "id",
  window_size = 12
) |>
  
  set_engine(
    "window_function",
    window_function = mean
  ) |>
  
  fit(
    value ~ .,
    data = train_tbl
  )


model_snaive_fit <- naive_reg(
  seasonal_period = 12,
  id = "id"
) |>
  
  set_engine(
    "snaive"
  ) |>
  
  fit(
    value ~ .,
    data = train_tbl
  )


# 25 - Elastic Net -------------------------------------------------------------

model_spec_glmnet <- linear_reg(
  penalty = 0.01,
  mixture = 0.99
) |>
  
  set_engine(
    "glmnet"
  )


wflw_fit_glmnet_lag <- workflow() |>
  add_model(model_spec_glmnet) |>
  add_recipe(recipe_spec_lag) |>
  fit(train_tbl) |>
  recursive(
    id = "id",
    transform = lag_roll_transformer_grouped,
    train_tail = panel_tail(
      train_tbl,
      id,
      FORECAST_HORIZON
    )
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
    train_tail = panel_tail(
      train_tbl,
      id,
      FORECAST_HORIZON
    )
  )


# 26 - ETS ---------------------------------------------------------------------

model_spec_ets <- exp_smoothing() |>
  set_engine(
    "ets"
  )


wflw_fit_ets <- workflow() |>
  add_model(model_spec_ets) |>
  add_recipe(recipe_spec_date) |>
  fit(train_tbl)


# 27 - TBATS -------------------------------------------------------------------

model_spec_tbats <- seasonal_reg(
  seasonal_period_1 = 12
) |>
  set_engine(
    "tbats"
  )


wflw_fit_tbats <- workflow() |>
  add_model(model_spec_tbats) |>
  add_recipe(recipe_spec_date) |>
  fit(train_tbl)


# 28 - STLM ETS ----------------------------------------------------------------

model_spec_stlm_ets <- seasonal_reg() |>
  set_engine(
    "stlm_ets"
  )


wflw_fit_stlm_ets <- workflow() |>
  add_model(model_spec_stlm_ets) |>
  add_recipe(recipe_spec_date) |>
  fit(train_tbl)


# 29 - ARIMA -------------------------------------------------------------------

model_spec_arima <- arima_reg() |>
  set_engine(
    "auto_arima"
  )


wflw_fit_arima <- workflow() |>
  add_model(model_spec_arima) |>
  add_recipe(recipe_spec_date) |>
  fit(train_tbl)


# 30 - NNETAR ------------------------------------------------------------------

model_spec_nnetar <- nnetar_reg() |>
  set_engine(
    "nnetar"
  )


wflw_fit_nnetar <- workflow() |>
  add_model(model_spec_nnetar) |>
  add_recipe(recipe_spec_date) |>
  fit(train_tbl)


# 31 - Prophet Boost -----------------------------------------------------------

model_spec_prophet_boost <- prophet_boost() |>
  set_engine(
    "prophet_xgboost"
  )


wflw_fit_prophet_boost <- workflow() |>
  add_model(model_spec_prophet_boost) |>
  add_recipe(recipe_spec_no_lag) |>
  fit(train_tbl)


# 32 - ARIMA Boost -------------------------------------------------------------

model_spec_arima_boost <- arima_boost() |>
  set_engine(
    "auto_arima_xgboost"
  )


wflw_fit_arima_boost <- workflow() |>
  add_model(model_spec_arima_boost) |>
  add_recipe(recipe_spec_no_lag) |>
  fit(train_tbl)


# 33 - XGBoost -----------------------------------------------------------------

model_spec_xgboost <- boost_tree(
  mode = "regression",
  trees = 500,
  learn_rate = 0.05,
  min_n = 5,
  tree_depth = 8,
  loss_reduction = 0.001
) |>
  
  set_engine(
    "xgboost"
  )


wflw_fit_xgboost_lag <- workflow() |>
  add_model(model_spec_xgboost) |>
  add_recipe(recipe_spec_lag) |>
  fit(train_tbl) |>
  recursive(
    id = "id",
    transform = lag_roll_transformer_grouped,
    train_tail = panel_tail(
      train_tbl,
      id,
      FORECAST_HORIZON
    )
  )


wflw_fit_xgboost_calendar <- workflow() |>
  add_model(model_spec_xgboost) |>
  add_recipe(recipe_spec_calendar) |>
  fit(train_tbl)


wflw_fit_xgboost_hybrid <- workflow() |>
  add_model(model_spec_xgboost) |>
  add_recipe(recipe_spec_hybrid) |>
  fit(train_tbl) |>
  recursive(
    id = "id",
    transform = lag_roll_transformer_grouped,
    train_tail = panel_tail(
      train_tbl,
      id,
      FORECAST_HORIZON
    )
  )


# 34 - LightGBM Base Models ----------------------------------------------------

model_spec_lightgbm <- boost_tree(
  mode = "regression",
  trees = 500,
  learn_rate = 0.05,
  tree_depth = 6,
  min_n = 5
) |>
  
  set_engine(
    "lightgbm",
    num_threads = 2
  )


wflw_fit_lightgbm_lag <- workflow() |>
  add_model(model_spec_lightgbm) |>
  add_recipe(recipe_spec_lag) |>
  fit(train_tbl) |>
  recursive(
    id = "id",
    transform = lag_roll_transformer_grouped,
    train_tail = panel_tail(
      train_tbl,
      id,
      FORECAST_HORIZON
    )
  )


wflw_fit_lightgbm_calendar <- workflow() |>
  add_model(model_spec_lightgbm) |>
  add_recipe(recipe_spec_calendar) |>
  fit(train_tbl)


wflw_fit_lightgbm_hybrid <- workflow() |>
  add_model(model_spec_lightgbm) |>
  add_recipe(recipe_spec_hybrid) |>
  fit(train_tbl) |>
  recursive(
    id = "id",
    transform = lag_roll_transformer_grouped,
    train_tail = panel_tail(
      train_tbl,
      id,
      FORECAST_HORIZON
    )
  )


# 35 - Random Forest -----------------------------------------------------------

model_spec_rf <- rand_forest(
  mode = "regression",
  trees = 500,
  min_n = 5
) |>
  
  set_engine(
    "randomForest"
  )


wflw_fit_rf_lag <- workflow() |>
  add_model(model_spec_rf) |>
  add_recipe(recipe_spec_lag) |>
  fit(train_tbl) |>
  recursive(
    id = "id",
    transform = lag_roll_transformer_grouped,
    train_tail = panel_tail(
      train_tbl,
      id,
      FORECAST_HORIZON
    )
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
    train_tail = panel_tail(
      train_tbl,
      id,
      FORECAST_HORIZON
    )
  )


# 36 - Support Vector Machine --------------------------------------------------

model_spec_svm <- svm_rbf(
  mode = "regression",
  margin = 0.001
) |>
  
  set_engine(
    "kernlab"
  )


wflw_fit_svm_lag <- workflow() |>
  add_model(model_spec_svm) |>
  add_recipe(recipe_spec_lag) |>
  fit(train_tbl) |>
  recursive(
    id = "id",
    transform = lag_roll_transformer_grouped,
    train_tail = panel_tail(
      train_tbl,
      id,
      FORECAST_HORIZON
    )
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
    train_tail = panel_tail(
      train_tbl,
      id,
      FORECAST_HORIZON
    )
  )


# 37 - DeepAR ------------------------------------------------------------------

if (RUN_DEEP_LEARNING) {
  
  set.seed(SEED)
  
  model_spec_deepar <- deep_ar(
    id = "id",
    freq = "M",
    prediction_length = 6,
    lookback_length = 18,
    epochs = 10
  ) |>
    
    set_engine(
      "gluonts_deepar"
    )
  
  
  wflw_fit_deepar <- workflow() |>
    add_model(model_spec_deepar) |>
    add_recipe(recipe_spec_no_lag) |>
    fit(train_tbl)
  
}


# 38 - Deep State --------------------------------------------------------------

if (RUN_DEEP_LEARNING) {
  
  set.seed(SEED)
  
  model_spec_deep_state <- deep_state(
    id = "id",
    freq = "M",
    prediction_length = 6,
    lookback_length = 18,
    epochs = 20
  ) |>
    
    set_engine(
      "gluonts_deepstate"
    )
  
  
  wflw_fit_deep_state <- workflow() |>
    add_model(model_spec_deep_state) |>
    add_recipe(recipe_spec_no_lag) |>
    fit(train_tbl)
  
}


# 39 - GP Forecaster -----------------------------------------------------------

if (RUN_DEEP_LEARNING) {
  
  set.seed(SEED)
  
  model_spec_gp <- gp_forecaster(
    id = "id",
    freq = "M",
    prediction_length = 6,
    epochs = 30
  ) |>
    
    set_engine(
      "gluonts_gp_forecaster"
    )
  
  
  wflw_fit_gp <- workflow() |>
    add_model(model_spec_gp) |>
    add_recipe(recipe_spec_no_lag) |>
    fit(train_tbl)
  
}


# 40 - Time-Series Cross Validation -------------------------------------------

tscv <- time_series_cv(
  data = train_tbl,
  date_var = date,
  cumulative = TRUE,
  initial = "60 months",
  assess = "24 months",
  skip = "6 months",
  slice_limit = 10
)


tscv |>
  tk_time_series_cv_plan() |>
  plot_time_series_cv_plan(
    .date_var = date,
    .value = value
  )


# 41 - Optional LightGBM Hyperparameter Tuning --------------------------------

if (RUN_TUNING) {
  
  # IMPORTANT:
  # Sequential processing is safer on Posit Cloud.
  plan(sequential)
  
  
  # Tune only learning rate and tree depth.
  # Trees and min_n remain fixed to reduce memory usage.
  
  lightgbm_tune_spec <- boost_tree(
    
    mode = "regression",
    
    trees = 300,
    
    learn_rate = tune(),
    
    tree_depth = tune(),
    
    min_n = 5
    
  ) |>
    
    set_engine(
      "lightgbm",
      num_threads = 1
    )
  
  
  lightgbm_tune_workflow <- workflow() |>
    
    add_model(
      lightgbm_tune_spec
    ) |>
    
    add_recipe(
      recipe_spec_calendar
    )
  
  
  # Smaller CV specifically for tuning
  
  tscv_tune <- time_series_cv(
    
    data = train_tbl,
    
    date_var = date,
    
    cumulative = TRUE,
    
    initial = "48 months",
    
    assess = "12 months",
    
    skip = "12 months",
    
    slice_limit = 4
  )
  
  
  set.seed(SEED)
  
  
  lightgbm_tune_results <- lightgbm_tune_workflow |>
    
    tune_grid(
      
      resamples = tscv_tune,
      
      grid = 6,
      
      metrics = metric_set(
        rmse,
        mae
      ),
      
      control = control_grid(
        verbose = TRUE,
        save_pred = FALSE,
        allow_par = FALSE
      )
    )
  
  
  print(
    show_best(
      lightgbm_tune_results,
      metric = "rmse"
    )
  )
  
  
  best_lightgbm_params <- lightgbm_tune_results |>
    
    select_best(
      metric = "rmse"
    )
  
  
  print(
    best_lightgbm_params
  )
  
  
  lightgbm_tuned_workflow <- lightgbm_tune_workflow |>
    
    finalize_workflow(
      best_lightgbm_params
    ) |>
    
    fit(
      train_tbl
    )
  
  
  lightgbm_tuned_workflow
  
}


# 42 - Build Global Model List -------------------------------------------------

global_model_list <- list(
  
  model_mean_fit_6,
  model_mean_fit_12,
  model_snaive_fit,
  
  wflw_fit_glmnet_lag,
  wflw_fit_glmnet_calendar,
  wflw_fit_glmnet_hybrid,
  
  wflw_fit_ets,
  wflw_fit_tbats,
  wflw_fit_stlm_ets,
  wflw_fit_arima,
  wflw_fit_nnetar,
  
  wflw_fit_prophet_boost,
  wflw_fit_arima_boost,
  
  wflw_fit_xgboost_lag,
  wflw_fit_xgboost_calendar,
  wflw_fit_xgboost_hybrid,
  
  wflw_fit_lightgbm_lag,
  wflw_fit_lightgbm_calendar,
  wflw_fit_lightgbm_hybrid,
  
  wflw_fit_rf_lag,
  wflw_fit_rf_calendar,
  wflw_fit_rf_hybrid,
  
  wflw_fit_svm_lag,
  wflw_fit_svm_calendar,
  wflw_fit_svm_hybrid
)


# Add tuned LightGBM to the candidate pool

if (RUN_TUNING) {
  
  global_model_list <- c(
    
    global_model_list,
    
    list(
      lightgbm_tuned_workflow
    )
  )
  
}


# Add deep-learning models

if (RUN_DEEP_LEARNING) {
  
  global_model_list <- c(
    
    global_model_list,
    
    list(
      wflw_fit_deepar,
      wflw_fit_deep_state,
      wflw_fit_gp
    )
  )
  
}


# 43 - Global Model Table ------------------------------------------------------

global_models_tbl <- do.call(
  
  modeltime_table,
  
  global_model_list
)


# Give tuned model a clear name

if (RUN_TUNING) {
  
  tuned_model_id <- max(
    global_models_tbl$.model_id
  )
  
  
  global_models_tbl <- global_models_tbl |>
    
    update_model_description(
      tuned_model_id,
      "Tuned LightGBM - Calendar"
    )
  
}


global_models_tbl


# 44 - Global Calibration ------------------------------------------------------

global_calibration_tbl <- global_models_tbl |>
  
  modeltime_calibrate(
    
    new_data = test_tbl,
    
    id = "id"
  )


# 45 - Global Accuracy ---------------------------------------------------------

global_accuracy_tbl <- global_calibration_tbl |>
  
  modeltime_accuracy(
    
    acc_by_id = FALSE
    
  ) |>
  
  arrange(
    rmse
  )


global_accuracy_tbl


# 46 - Accuracy by Energy Series ----------------------------------------------

global_accuracy_by_id_tbl <- global_calibration_tbl |>
  
  modeltime_accuracy(
    
    acc_by_id = TRUE
    
  ) |>
  
  group_by(
    id
  ) |>
  
  arrange(
    rmse
  )


global_accuracy_by_id_tbl 


# 47 - Top Five Global Models --------------------------------------------------

top_global_models <- global_accuracy_tbl |>
  
  slice_min(
    
    order_by = rmse,
    
    n = 5
  )


top_global_models


# 48 - Global Forecast Comparison ---------------------------------------------

top_model_ids <- top_global_models |>
  
  pull(
    .model_id
  )


global_calibration_tbl |>
  
  filter(
    .model_id %in% top_model_ids
  ) |>
  
  modeltime_forecast(
    
    new_data = test_tbl,
    
    actual_data = data_prepared_tbl,
    
    keep_data = TRUE,
    
    conf_by_id = TRUE
  ) |>
  
  group_by(
    id
  ) |>
  
  plot_modeltime_forecast(
    
    .facet_ncol = 2,
    
    .trelliscope = FALSE,
    
    .interactive = FALSE
  )


# 49 - Ensemble Candidate Selection -------------------------------------------

ensemble_candidate_ids <- global_accuracy_tbl |>
  
  slice_min(
    
    order_by = rmse,
    
    n = 4
    
  ) |>
  
  pull(
    .model_id
  )


ensemble_candidate_ids


ensemble_candidates <- global_calibration_tbl |>
  
  filter(
    .model_id %in% ensemble_candidate_ids
  )


# 50 - Mean Ensemble -----------------------------------------------------------

ensemble_mean <- ensemble_candidates |>
  
  ensemble_average(
    type = "mean"
  )


# 51 - Median Ensemble ---------------------------------------------------------

ensemble_median <- ensemble_candidates |>
  
  ensemble_average(
    type = "median"
  )


# 52 - Weighted Ensemble -------------------------------------------------------

number_ensemble_models <- nrow(
  ensemble_candidates
)


ensemble_weights <- rev(
  seq_len(
    number_ensemble_models
  )
)


ensemble_weighted_model <- ensemble_candidates |>
  
  ensemble_weighted(
    loadings = ensemble_weights
  )


# 53 - Ensemble Calibration ----------------------------------------------------

ensemble_models_tbl <- modeltime_table(
  
  ensemble_mean,
  ensemble_median,
  ensemble_weighted_model
  
) |>
  
  update_model_description(
    1,
    "Mean Ensemble"
  ) |>
  
  update_model_description(
    2,
    "Median Ensemble"
  ) |>
  
  update_model_description(
    3,
    "Weighted Ensemble"
  )


ensemble_calibration_tbl <- ensemble_models_tbl |>
  
  modeltime_calibrate(
    
    test_tbl,
    
    id = "id"
  )


ensemble_accuracy_tbl <- ensemble_calibration_tbl |>
  
  modeltime_accuracy(
    
    acc_by_id = FALSE
    
  ) |>
  
  arrange(
    rmse
  )


ensemble_accuracy_tbl

# 54 - Optional Stacked Ensemble ----------------------------------------------

if (RUN_STACKING) {
  
 
  # Deep-learning models remain in the final benchmark,
  # but are excluded from modeltime stacking.
  #
  # Stacking is performed only with models that are stable under
  # tidymodels / modeltime rolling resampling.
  
  stack_eligible_accuracy <- global_accuracy_tbl |>
    
    filter(
      !str_detect(
        .model_desc,
        regex(
          "DEEPAR|DEEPSTATE|GP FORECASTER",
          ignore_case = TRUE
        )
      )
    )
  
  
  # Select the three best stack-compatible models
  
  stack_base_ids <- stack_eligible_accuracy |>
    
    slice_min(
      order_by = rmse,
      n = 3
    ) |>
    
    pull(
      .model_id
    )
  
  
  print(
    stack_eligible_accuracy |>
      filter(
        .model_id %in% stack_base_ids
      ) |>
      select(
        .model_id,
        .model_desc,
        rmse,
        mae,
        mape
      )
  )
  
  
  # Extract those models from the original model table
  
  stack_base_models <- global_models_tbl |>
    
    filter(
      .model_id %in% stack_base_ids
    )
  
  
  print(
    stack_base_models
  )
  
  

  # Rolling resamples
  
  
  stack_resamples <- time_series_cv(
    
    data = train_tbl,
    
    date_var = date,
    
    cumulative = TRUE,
    
    initial = "48 months",
    
    assess = "12 months",
    
    skip = "12 months",
    
    slice_limit = 4
  )
  
  
  # Generate out-of-sample predictions for base models
  
  set.seed(SEED)

  stack_resample_fits <- stack_base_models |>
    
    modeltime_fit_resamples(
      
      resamples = stack_resamples,
      
      control = control_resamples(
        
        verbose = TRUE,
        
        save_pred = TRUE,
        
        allow_par = FALSE
      )
    )
  
  
  # Check resampling performance BEFORE attempting stacking
  
  stack_resample_accuracy <- stack_resample_fits |>
    
    modeltime_resample_accuracy(
      
      metric_set = metric_set(
        rmse,
        mae
      )
    )
  
  
  print(
    stack_resample_accuracy
  )
  
  
# Elastic Net meta learner
 
  
  stack_meta_spec <- linear_reg(
    
    penalty = tune(),
    
    mixture = tune()
    
  ) |>
    
    set_engine(
      "glmnet"
    )
  
  
  # Stacked Ensemble
  
  set.seed(SEED)
  
  stacked_ensemble <- stack_resample_fits |>
    
    ensemble_model_spec(
      
      model_spec = stack_meta_spec,
      
      kfolds = 4,
      
      grid = 6,
      
      control = control_grid(
        
        verbose = TRUE,
        
        allow_par = FALSE
      )
    )
  
  
  # Modeltime table
  
  stacked_tbl <- modeltime_table(
    
    stacked_ensemble
    
  ) |>
    
    update_model_description(
      
      1,
      
      "Stacked Ensemble - Elastic Net"
    )
  
  
  # Holdout calibration

  stacked_calibration_tbl <- stacked_tbl |>
    
    modeltime_calibrate(
      
      test_tbl,
      
      id = "id"
    )
  
  
  stacked_accuracy_tbl <- stacked_calibration_tbl |>
    
    modeltime_accuracy(
      
      acc_by_id = FALSE
      
    ) |>
    
    arrange(
      rmse
    )
  
  
  print(
    stacked_accuracy_tbl
  )
}

# 55 - Combine Global and Ensemble Results ------------------------------------

all_models_tbl <- combine_modeltime_tables(
  
  global_models_tbl,
  
  ensemble_models_tbl
  
) |>
  
  modeltime_calibrate(
    
    test_tbl,
    
    id = "id"
  )


if (RUN_STACKING) {
  
  all_models_tbl <- combine_modeltime_tables(
    
    all_models_tbl,
    
    stacked_tbl
    
  ) |>
    
    modeltime_calibrate(
      
      test_tbl,
      
      id = "id"
    )
  
}


# 56 - Final Model Accuracy ----------------------------------------------------

all_models_accuracy_tbl <- all_models_tbl |>
  
  modeltime_accuracy(
    
    acc_by_id = FALSE
    
  ) |>
  
  arrange(
    rmse
  )


all_models_accuracy_tbl


# 57 - Final Accuracy Chart ----------------------------------------------------

all_models_accuracy_tbl |>
  
  mutate(
    model_label = paste0(
      .model_desc,
      " [", .model_id, "]"
    )
  ) |>
  
  arrange(
    rmse
  ) |>
  
  mutate(
    model_label = factor(
      model_label,
      levels = rev(model_label)
    )
  ) |>
  
  ggplot(
    aes(
      x = rmse,
      y = model_label
    )
  ) +
  
  geom_col() +
  
  labs(
    title = "Forecast Model Comparison",
    subtitle = "Models ranked from lowest to highest RMSE",
    x = "RMSE",
    y = NULL
  ) +
  
  theme_minimal()


# 58 - Best Global Model -------------------------------------------------------

best_global_model_id <- all_models_accuracy_tbl |>
  
  slice_min(
    
    order_by = rmse,
    
    n = 1,
    
    with_ties = FALSE
    
  ) |>
  
  pull(
    .model_id
  )


best_global_model <- all_models_tbl |>
  
  filter(
    .model_id == best_global_model_id
  )


best_global_model


# 59 - Best Global Accuracy by Series -----------------------------------------

best_global_local_accuracy <- best_global_model |>
  
  modeltime_calibrate(
    
    test_tbl,
    
    id = "id"
    
  ) |>
  
  modeltime_accuracy(
    
    acc_by_id = TRUE
    
  ) |>
  
  select(
    
    id,
    
    global_rmse = rmse,
    
    global_mae = mae,
    
    global_mape = mape
  )


best_global_local_accuracy

# clear some large objects you no longer need

#rm(
#  lightgbm_tune_results,
#  tscv,
 # tscv_tune,
#  stack_resample_fits
#)

#gc()


# 60 - Nested Series-Specific Setup -------------------------------------------

if (RUN_NESTED_MODELS) {
  
  nested_data_tbl <- model_data |>
    
    nest_timeseries(
      
      .id_var = id,
      
      .length_future = FORECAST_HORIZON
      
    ) |>
    
    split_nested_timeseries(
      
      .length_test = FORECAST_HORIZON
    )
  
  
  nested_train_tbl <- extract_nested_train_split(
    
    nested_data_tbl
  )
  
}


# 61 - Local Model Recipes -----------------------------------------------------

if (RUN_NESTED_MODELS) {
  
  local_recipe_date <- recipe(
    
    value ~ .,
    
    data = nested_train_tbl |>
      
      select(
        date,
        value
      )
  )
  
  
  local_recipe_calendar <- recipe(
    
    value ~ .,
    
    data = nested_train_tbl
    
  ) |>
    
    step_timeseries_signature(
      date
    ) |>
    
    step_rm(
      contains("iso"),
      contains("xts")
    ) |>
    
    step_dummy(
      all_nominal_predictors(),
      one_hot = TRUE
    ) |>
    
    step_normalize(
      date_index.num,
      starts_with("date_year")
    ) |>
    
    step_rm(
      date
    ) |>
    
    step_zv(
      all_predictors()
    )
  
}

# 62 - Local Candidate Models --------------------------------------------------

if (RUN_NESTED_MODELS) {
  
  local_snaive <- workflow() |>
    
    add_model(
      naive_reg(
        seasonal_period = 12
      ) |>
        set_engine("snaive")
    ) |>
    
    add_recipe(
      local_recipe_date
    )
  
  
  local_ets <- workflow() |>
    
    add_model(
      exp_smoothing() |>
        set_engine("ets")
    ) |>
    
    add_recipe(
      local_recipe_date
    )
  
  
  local_arima <- workflow() |>
    
    add_model(
      arima_reg() |>
        set_engine("auto_arima")
    ) |>
    
    add_recipe(
      local_recipe_date
    )
  
  
  local_xgb <- workflow() |>
    
    add_model(
      boost_tree(
        mode = "regression",
        trees = 200,
        learn_rate = 0.05,
        tree_depth = 5,
        min_n = 5
      ) |>
        set_engine(
          "xgboost",
          nthread = 1
        )
    ) |>
    
    add_recipe(
      local_recipe_calendar
    )
}

# 63 - Fit Nested Candidate Models --------------------------------------------

if (RUN_NESTED_MODELS) {
  
  nested_modeltime_tbl <- nested_data_tbl |>
    
    modeltime_nested_fit(
      
      model_list = list(
        local_snaive,
        local_ets,
        local_arima,
        local_xgb
      ),
      
      control = control_nested_fit(
        verbose = TRUE,
        allow_par = FALSE
      )
    )
  
  nested_modeltime_tbl
}


# 64 - Best Series-Specific Model ---------------------------------------------

if (RUN_NESTED_MODELS) {
  
  best_local_models <- nested_modeltime_tbl |>
    
    modeltime_nested_select_best(
      
      metric = "rmse",
      
      minimize = TRUE,
      
      filter_test_forecasts = TRUE
    )
  
  
  local_model_accuracy <- best_local_models |>
    
    extract_nested_best_model_report()
  
  
  local_model_accuracy
  
}


# 65 - Global vs Series-Specific Comparison -----------------------------------

if (RUN_NESTED_MODELS) {
  
  global_vs_local <- best_global_local_accuracy |>
    
    inner_join(
      
      local_model_accuracy |>
        
        select(
          
          id,
          
          local_model = .model_desc,
          
          local_rmse = rmse,
          
          local_mae = mae,
          
          local_mape = mape
          
        ),
      
      by = "id"
      
    ) |>
    
    mutate(
      
      winner = case_when(
        
        global_rmse < local_rmse ~ "Global",
        
        local_rmse < global_rmse ~ "Series-Specific",
        
        TRUE ~ "Tie"
      ),
      
      rmse_improvement =
        global_rmse -
        local_rmse,
      
      rmse_improvement_pct =
        100 *
        (
          global_rmse -
            local_rmse
        ) /
        global_rmse
    )
  
  
  global_vs_local
  
  
  global_vs_local |>
    
    count(
      winner,
      sort = TRUE
    )
  
}




# 66 - Global vs Series-Specific Chart ----------------------------------------

global_vs_local_plot <- global_vs_local |>
  
  select(
    id,
    global_rmse,
    local_rmse
  ) |>
  
  mutate(
    rmse_gap = abs(global_rmse - local_rmse)
  ) |>
  
  arrange(rmse_gap) |>
  
  mutate(
    id = factor(
      id,
      levels = id
    )
  ) |>
  
  pivot_longer(
    cols = c(
      global_rmse,
      local_rmse
    ),
    names_to = "strategy",
    values_to = "rmse"
  ) |>
  
  mutate(
    strategy = recode(
      strategy,
      global_rmse = "Global",
      local_rmse = "Series-Specific"
    )
  )


ggplot(
  global_vs_local_plot,
  aes(
    x = rmse,
    y = id,
    fill = strategy
  )
) +
  
  geom_col(
    position = position_dodge(width = 0.9),
    width = 0.85
  ) +
  
  geom_text(
    aes(
      label = round(rmse, 3)
    ),
    position = position_dodge(width = 0.9),
    hjust = -0.12,
    size = 3
  ) +
  
  labs(
    title = "Global vs Series-Specific Forecasting",
    subtitle = "24-month holdout RMSE by energy series — lower is better",
    x = "RMSE",
    y = NULL,
    fill = "Strategy"
  ) +
  
  expand_limits(
    x = max(global_vs_local_plot$rmse) * 1.08
  ) +
  
  theme_minimal() +
  
  theme(
    plot.title = element_text(
      size = 15,
      face = "bold"
    ),
    plot.subtitle = element_text(
      size = 11
    ),
    axis.text.y = element_text(
      size = 9
    ),
    legend.position = "right"
  )

# Global vs Local Summary -----------------------------------------------------

global_vs_local |>
  
  select(
    id,
    global_rmse,
    local_model,
    local_rmse,
    winner,
    rmse_improvement_pct
  ) |>
  
  arrange(
    global_rmse
  )


global_vs_local |>
  count(
    winner,
    sort = TRUE
  )


if (RUN_NESTED_MODELS) {
  
  best_local_refit <- best_local_models |>
    
    modeltime_nested_refit(
      
      control = control_refit(
        
        verbose = TRUE,
        
        allow_par = FALSE
      )
    )
  
  
  best_local_refit |>
    
    extract_nested_error_report()
  
}


# 68 - Final Local 24-Month Forecast ------------------------------------------

if (RUN_NESTED_MODELS) {
  
  local_future_forecast <- best_local_refit |>
    
    extract_nested_future_forecast()
  
  
  local_future_forecast |>
    
    group_by(
      id
    ) |>
    
    plot_modeltime_forecast(
      
      .facet_ncol = 2,
      
      .trelliscope = FALSE,
      
      .interactive = FALSE,
      
      .legend_show = FALSE
    )
  
}


# 69 - Refit Best Global Model -------------------------------------------------

best_global_refit <- best_global_model |>
  
  modeltime_refit(
    
    data = data_prepared_tbl,
    
    control = control_refit(
      verbose = TRUE
    )
  )


best_global_refit


# 70 - Final Global 24-Month Forecast -----------------------------------------

global_future_forecast <- best_global_refit |>
  
  modeltime_forecast(
    
    new_data = data_future_tbl,
    
    actual_data = data_prepared_tbl,
    
    keep_data = TRUE,
    
    conf_by_id = TRUE
  )


global_future_forecast |>
  
  group_by(
    id
  ) |>
  
  plot_modeltime_forecast(
    
    .facet_ncol = 2,
    
    .trelliscope = FALSE,
    
    .interactive = FALSE
  )


# 71 - Foundation Model Data ---------------------------------------------------

if (RUN_FOUNDATION_MODELS) {
  
  dir.create(
    "data/foundation",
    recursive = TRUE,
    showWarnings = FALSE
  )
  
  # Use exactly the same holdout period as the R / modeltime workflow
  
  foundation_test_start <- min(
    test_tbl$date
  )
  
  
  foundation_context <- model_data |>
    
    dplyr::filter(
      date < foundation_test_start
    ) |>
    
    dplyr::arrange(
      id,
      date
    )
  
  
  # Use exactly the same observations used for modeltime testing
  
  foundation_truth <- test_tbl |>
    
    dplyr::select(
      id,
      date,
      value
    ) |>
    
    dplyr::arrange(
      id,
      date
    )
  
  
  readr::write_csv(
    foundation_context,
    "data/foundation/context.csv"
  )
  
  
  readr::write_csv(
    foundation_truth,
    "data/foundation/truth.csv"
  )
  
  
  print(
    foundation_context |>
      dplyr::group_by(id) |>
      dplyr::summarise(
        context_rows = dplyr::n(),
        first_date = min(date),
        last_date = max(date),
        .groups = "drop"
      )
  )
  
  
  print(
    foundation_truth |>
      dplyr::group_by(id) |>
      dplyr::summarise(
        test_rows = dplyr::n(),
        first_test_date = min(date),
        last_test_date = max(date),
        .groups = "drop"
      )
  )
  
}


# 72 - Foundation Model Python Environment ------------------------------------

if (RUN_FOUNDATION_MODELS) {
  
  library(reticulate)
  
  use_condaenv(
    "energy-timesfm",
    required = TRUE
  )
  
  
  py_config()
  
  
  if (!py_module_available("chronos")) {
    
    stop(
      "Chronos is not installed in the 'energy-timesfm' Conda environment."
    )
  }
  
  
  if (!py_module_available("timesfm")) {
    
    stop(
      "TimesFM is not installed in the 'energy-timesfm' Conda environment."
    )
  }
  
  
  message(
    "Foundation-model Python environment is ready."
  )
  
}


# 73 - Chronos-2 Zero-Shot Forecast -------------------------------------------

if (RUN_FOUNDATION_MODELS) {
  
  py_run_string("
import pandas as pd
from chronos import Chronos2Pipeline

FORECAST_HORIZON = 24

context = pd.read_csv(
    'data/foundation/context.csv',
    parse_dates=['date']
)

pipeline = Chronos2Pipeline.from_pretrained(
    'amazon/chronos-2',
    device_map='cpu'
)

chronos_forecast = pipeline.predict_df(
    context,
    prediction_length=FORECAST_HORIZON,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column='id',
    timestamp_column='date',
    target='value',
    freq='MS'
)

chronos_forecast.to_csv(
    'outputs/chronos2_forecast.csv',
    index=False
)

print(chronos_forecast.head())
print('Chronos-2 rows written:', len(chronos_forecast))
")

}


# 74 - Chronos-2 Accuracy ------------------------------------------------------

if (RUN_FOUNDATION_MODELS) {
  
  if (!file.exists(
    "outputs/chronos2_forecast.csv"
  )) {
    
    stop(
      "Chronos forecast file was not created."
    )
  }
  
  
  chronos_forecast_r <- readr::read_csv(
    "outputs/chronos2_forecast.csv",
    show_col_types = FALSE
  ) |>
    
    dplyr::mutate(
      date = as.Date(date)
    )
  
  
  print(
    names(chronos_forecast_r)
  )
  
  
  # Detect Chronos point forecast column
  
  chronos_prediction_col <- dplyr::case_when(
    
    "predictions" %in%
      names(chronos_forecast_r) ~ "predictions",
    
    "prediction" %in%
      names(chronos_forecast_r) ~ "prediction",
    
    "0.5" %in%
      names(chronos_forecast_r) ~ "0.5",
    
    TRUE ~ NA_character_
  )
  
  
  if (is.na(
    chronos_prediction_col
  )) {
    
    stop(
      paste(
        "Could not identify Chronos point forecast column.",
        "Available columns:",
        paste(
          names(chronos_forecast_r),
          collapse = ", "
        )
      )
    )
  }
  
  
  chronos_forecast_r <- chronos_forecast_r |>
    
    dplyr::rename(
      chronos_prediction =
        dplyr::all_of(
          chronos_prediction_col
        )
    )
  
  
  chronos_accuracy_by_id <- foundation_truth |>
    
    dplyr::inner_join(
      chronos_forecast_r,
      by = c(
        "id",
        "date"
      )
    ) |>
    
    dplyr::group_by(
      id
    ) |>
    
    dplyr::summarise(
      
      rmse = yardstick::rmse_vec(
        truth = value,
        estimate = chronos_prediction
      ),
      
      mae = yardstick::mae_vec(
        truth = value,
        estimate = chronos_prediction
      ),
      
      mape = yardstick::mape_vec(
        truth = value,
        estimate = chronos_prediction
      ),
      
      .groups = "drop"
    )
  
  
  chronos_accuracy <- foundation_truth |>
    
    dplyr::inner_join(
      chronos_forecast_r,
      by = c(
        "id",
        "date"
      )
    ) |>
    
    dplyr::summarise(
      
      .model_desc =
        "Chronos-2 Zero-Shot",
      
      rmse = yardstick::rmse_vec(
        truth = value,
        estimate = chronos_prediction
      ),
      
      mae = yardstick::mae_vec(
        truth = value,
        estimate = chronos_prediction
      ),
      
      mape = yardstick::mape_vec(
        truth = value,
        estimate = chronos_prediction
      )
    )
  
  
  print(
    chronos_accuracy
  )
  
  
  print(
    chronos_accuracy_by_id |>
      dplyr::arrange(rmse)
  )
  
}


# 75 - TimesFM 3.0 Zero-Shot Forecast -----------------------------------------

if (RUN_FOUNDATION_MODELS) {
  
  py_run_string("
import pandas as pd
import numpy as np

from timesfm3 import TimesFM3Evaluator, ModelConfig

FORECAST_HORIZON = 24

context = pd.read_csv(
    'data/foundation/context.csv',
    parse_dates=['date']
)

config = ModelConfig(
    checkpoint_path='google/timesfm-3.0-pytorch',
    per_core_batch_size=16,
    device='cpu'
)

forecaster = TimesFM3Evaluator(
    config
)

forecast_rows = []

for series_id, df_series in context.groupby('id'):

    df_series = df_series.sort_values(
        'date'
    )

    values = df_series[
        'value'
    ].to_numpy(
        dtype=np.float32
    )

    outputs = list(
        forecaster.predict_batch(
            [values],
            horizon=FORECAST_HORIZON,
            return_quantiles=True,
            use_symmetric_averaging=False
        )
    )

    prediction = outputs[0].forecast

    last_date = df_series[
        'date'
    ].max()

    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=FORECAST_HORIZON,
        freq='MS'
    )

    for date_value, prediction_value in zip(
        future_dates,
        prediction
    ):

        forecast_rows.append({
            'id': series_id,
            'date': date_value,
            'timesfm_prediction':
                float(prediction_value)
        })

timesfm_forecast = pd.DataFrame(
    forecast_rows
)

timesfm_forecast.to_csv(
    'outputs/timesfm3_forecast.csv',
    index=False
)

print(timesfm_forecast.head())
print('TimesFM rows written:', len(timesfm_forecast))
")

}


# 76 - TimesFM 3.0 Accuracy ----------------------------------------------------

if (RUN_FOUNDATION_MODELS) {
  
  if (!file.exists(
    "outputs/timesfm3_forecast.csv"
  )) {
    
    stop(
      "TimesFM forecast file was not created."
    )
  }
  
  
  timesfm_forecast_r <- readr::read_csv(
    "outputs/timesfm3_forecast.csv",
    show_col_types = FALSE
  ) |>
    
    dplyr::mutate(
      date = as.Date(date)
    )
  
  
  timesfm_accuracy_by_id <- foundation_truth |>
    
    dplyr::inner_join(
      timesfm_forecast_r,
      by = c(
        "id",
        "date"
      )
    ) |>
    
    dplyr::group_by(
      id
    ) |>
    
    dplyr::summarise(
      
      rmse = yardstick::rmse_vec(
        truth = value,
        estimate = timesfm_prediction
      ),
      
      mae = yardstick::mae_vec(
        truth = value,
        estimate = timesfm_prediction
      ),
      
      mape = yardstick::mape_vec(
        truth = value,
        estimate = timesfm_prediction
      ),
      
      .groups = "drop"
    )
  
  
  timesfm_accuracy <- foundation_truth |>
    
    dplyr::inner_join(
      timesfm_forecast_r,
      by = c(
        "id",
        "date"
      )
    ) |>
    
    dplyr::summarise(
      
      .model_desc =
        "TimesFM 3.0 Zero-Shot",
      
      rmse = yardstick::rmse_vec(
        truth = value,
        estimate = timesfm_prediction
      ),
      
      mae = yardstick::mae_vec(
        truth = value,
        estimate = timesfm_prediction
      ),
      
      mape = yardstick::mape_vec(
        truth = value,
        estimate = timesfm_prediction
      )
    )
  
  
  print(
    timesfm_accuracy
  )
  
  
  print(
    timesfm_accuracy_by_id |>
      dplyr::arrange(rmse)
  )
  
}


# 77 - Foundation Model Comparison by Series ----------------------------------

if (RUN_FOUNDATION_MODELS) {
  
  foundation_accuracy_by_id <- chronos_accuracy_by_id |>
    
    dplyr::rename(
      chronos_rmse = rmse,
      chronos_mae = mae,
      chronos_mape = mape
    ) |>
    
    dplyr::inner_join(
      
      timesfm_accuracy_by_id |>
        
        dplyr::rename(
          timesfm_rmse = rmse,
          timesfm_mae = mae,
          timesfm_mape = mape
        ),
      
      by = "id"
    ) |>
    
    dplyr::mutate(
      
      foundation_winner =
        dplyr::case_when(
          
          chronos_rmse <
            timesfm_rmse ~ "Chronos-2",
          
          timesfm_rmse <
            chronos_rmse ~ "TimesFM 3.0",
          
          TRUE ~ "Tie"
        )
    )
  
  
  print(
    foundation_accuracy_by_id |>
      dplyr::arrange(
        pmin(
          chronos_rmse,
          timesfm_rmse
        )
      )
  )
  
  
  print(
    foundation_accuracy_by_id |>
      
      dplyr::count(
        foundation_winner,
        sort = TRUE
      )
  )
  
}


# 78 - Final Benchmark ---------------------------------------------------------

if (RUN_FOUNDATION_MODELS) {
  
  final_benchmark_tbl <- dplyr::bind_rows(
    
    all_models_accuracy_tbl |>
      
      dplyr::select(
        .model_desc,
        rmse,
        mae,
        mape
      ),
    
    chronos_accuracy,
    
    timesfm_accuracy
    
  ) |>
    
    dplyr::arrange(
      rmse
    )
  
  
  print(
    final_benchmark_tbl
  )
  
  
  print(
    final_benchmark_tbl |>
      dplyr::slice_head(
        n = 15
      )
  )
  
}


# 79 - Final Benchmark Chart ---------------------------------------------------

if (RUN_FOUNDATION_MODELS) {
  
  final_benchmark_plot <- final_benchmark_tbl |>
    
    dplyr::mutate(
      
      model_label =
        make.unique(
          .model_desc
        )
    ) |>
    
    dplyr::arrange(
      rmse
    ) |>
    
    dplyr::slice_head(
      n = 15
    ) |>
    
    dplyr::mutate(
      
      model_label = factor(
        model_label,
        levels = rev(
          model_label
        )
      )
    )
  
  
  ggplot(
    final_benchmark_plot,
    aes(
      x = rmse,
      y = model_label
    )
  ) +
    
    geom_col() +
    
    geom_text(
      
      aes(
        label = round(
          rmse,
          3
        )
      ),
      
      hjust = -0.15,
      
      size = 3
    ) +
    
    labs(
      
      title =
        "Final Forecasting Benchmark",
      
      subtitle = paste(
        "24-month holdout:",
        "classical, machine learning,",
        "deep learning, ensembles",
        "and zero-shot foundation models"
      ),
      
      x = "RMSE",
      
      y = NULL
    ) +
    
    expand_limits(
      
      x =
        max(
          final_benchmark_plot$rmse
        ) * 1.12
    ) +
    
    theme_minimal()
  
}

# 80 - Save Accuracy Tables ----------------------------------------------------

readr::write_csv(
  global_accuracy_tbl,
  "outputs/global_accuracy.csv"
)


readr::write_csv(
  global_accuracy_by_id_tbl,
  "outputs/global_accuracy_by_series.csv"
)


readr::write_csv(
  ensemble_accuracy_tbl,
  "outputs/ensemble_accuracy.csv"
)


readr::write_csv(
  all_models_accuracy_tbl,
  "outputs/all_models_accuracy.csv"
)


# Save LightGBM tuning results

if (RUN_TUNING) {
  
  lightgbm_tuning_metrics <- tune::collect_metrics(
    lightgbm_tune_results
  )
  
  
  readr::write_csv(
    lightgbm_tuning_metrics,
    "outputs/lightgbm_tuning_results.csv"
  )
  
  
  # Save selected best parameters as a separate file
  
  readr::write_csv(
    tibble::as_tibble(
      best_lightgbm_params
    ),
    "outputs/best_lightgbm_parameters.csv"
  )
}


# Save nested / local model results

if (RUN_NESTED_MODELS) {
  
  readr::write_csv(
    local_model_accuracy,
    "outputs/local_model_accuracy.csv"
  )
  
  
  readr::write_csv(
    global_vs_local,
    "outputs/global_vs_local.csv"
  )
}


# Save foundation model results

if (RUN_FOUNDATION_MODELS) {
  
  readr::write_csv(
    chronos_accuracy,
    "outputs/chronos2_accuracy.csv"
  )
  
  
  readr::write_csv(
    chronos_accuracy_by_id,
    "outputs/chronos2_accuracy_by_series.csv"
  )
  
  
  readr::write_csv(
    timesfm_accuracy,
    "outputs/timesfm3_accuracy.csv"
  )
  
  
  readr::write_csv(
    timesfm_accuracy_by_id,
    "outputs/timesfm3_accuracy_by_series.csv"
  )
  
  
  readr::write_csv(
    foundation_accuracy_by_id,
    "outputs/foundation_model_comparison_by_series.csv"
  )
  
  
  readr::write_csv(
    final_benchmark_tbl,
    "outputs/final_model_benchmark.csv"
  )
}


# 81 - Save Final Models -------------------------------------------------------

write_rds(
  best_global_refit,
  "models/best_global_model.rds"
)


if (RUN_TUNING) {
  
  write_rds(
    lightgbm_tuned_workflow,
    "models/tuned_lightgbm.rds"
  )
}


if (RUN_NESTED_MODELS) {
  
  write_rds(
    best_local_refit,
    "models/best_local_models.rds"
  )
}


# 82 - Save Final Forecasts ----------------------------------------------------

readr::write_csv(
  global_future_forecast,
  "outputs/global_future_forecast.csv"
)


if (RUN_NESTED_MODELS) {
  
  readr::write_csv(
    local_future_forecast,
    "outputs/local_future_forecast.csv"
  )
}


if (RUN_FOUNDATION_MODELS) {
  
  readr::write_csv(
    chronos_forecast_r,
    "outputs/chronos2_forecast_clean.csv"
  )
  
  
  readr::write_csv(
    timesfm_forecast_r,
    "outputs/timesfm3_forecast_clean.csv"
  )
}


# 83 - Final Benchmark Winner --------------------------------------------------

if (RUN_FOUNDATION_MODELS) {
  
  final_winner <- final_benchmark_tbl |>
    
    dplyr::slice_min(
      order_by = rmse,
      n = 1,
      with_ties = FALSE
    )
  
  
  cat(
    "\n============================================================\n"
  )
  
  cat(
    "FINAL BENCHMARK WINNER\n"
  )
  
  cat(
    "============================================================\n\n"
  )
  
  
  print(
    final_winner
  )
}


# 84 - Top 10 Final Models -----------------------------------------------------

if (RUN_FOUNDATION_MODELS) {
  
  top_10_models <- final_benchmark_tbl |>
    
    dplyr::slice_head(
      n = 10
    )
  
  
  cat(
    "\n============================================================\n"
  )
  
  cat(
    "TOP 10 FORECASTING MODELS\n"
  )
  
  cat(
    "============================================================\n\n"
  )
  
  
  print(
    top_10_models
  )
  
  
  readr::write_csv(
    top_10_models,
    "outputs/top_10_models.csv"
  )
}


# 85 - Global vs Series-Specific Summary --------------------------------------

if (RUN_NESTED_MODELS) {
  
  global_vs_local_summary <- global_vs_local |>
    
    dplyr::count(
      winner,
      sort = TRUE
    )
  
  
  cat(
    "\n============================================================\n"
  )
  
  cat(
    "GLOBAL VS SERIES-SPECIFIC RESULTS\n"
  )
  
  cat(
    "============================================================\n\n"
  )
  
  
  print(
    global_vs_local_summary
  )
  
  
  readr::write_csv(
    global_vs_local_summary,
    "outputs/global_vs_local_summary.csv"
  )
}


# 86 - Foundation Model Summary -----------------------------------------------

if (RUN_FOUNDATION_MODELS) {
  
  foundation_summary <- dplyr::bind_rows(
    
    chronos_accuracy,
    
    timesfm_accuracy
    
  ) |>
    
    dplyr::arrange(
      rmse
    )
  
  
  cat(
    "\n============================================================\n"
  )
  
  cat(
    "FOUNDATION MODEL RESULTS\n"
  )
  
  cat(
    "============================================================\n\n"
  )
  
  
  print(
    foundation_summary
  )
  
  
  readr::write_csv(
    foundation_summary,
    "outputs/foundation_model_summary.csv"
  )
}


# 87 - Foundation Model Winners by Series -------------------------------------

if (RUN_FOUNDATION_MODELS) {
  
  foundation_winner_summary <- foundation_accuracy_by_id |>
    
    dplyr::count(
      foundation_winner,
      sort = TRUE
    )
  
  
  print(
    foundation_winner_summary
  )
  
  
  readr::write_csv(
    foundation_winner_summary,
    "outputs/foundation_model_winners_by_series.csv"
  )
}


# 88 - Final Project Summary ---------------------------------------------------

cat(
  "\n\n",
  "============================================================\n",
  "ENERGY PRODUCTION FORECASTING - FINAL PROJECT SUMMARY\n",
  "============================================================\n"
)


cat(
  "\nBest R / Modeltime model:\n"
)


best_r_model <- all_models_accuracy_tbl |>
  
  dplyr::slice_min(
    order_by = rmse,
    n = 1,
    with_ties = FALSE
  )


print(
  best_r_model
)


if (RUN_FOUNDATION_MODELS) {
  
  cat(
    "\nBest model across the complete benchmark:\n"
  )
  
  
  print(
    final_benchmark_tbl |>
      
      dplyr::slice_min(
        order_by = rmse,
        n = 1,
        with_ties = FALSE
      )
  )
  
  
  cat(
    "\nTop 10 overall:\n"
  )
  
  
  print(
    final_benchmark_tbl |>
      
      dplyr::slice_head(
        n = 10
      )
  )
}


if (RUN_NESTED_MODELS) {
  
  cat(
    "\nGlobal vs series-specific winners:\n"
  )
  
  
  print(
    global_vs_local |>
      
      dplyr::count(
        winner,
        sort = TRUE
      )
  )
}


if (RUN_FOUNDATION_MODELS) {
  
  cat(
    "\nFoundation model comparison:\n"
  )
  
  
  print(
    foundation_summary
  )
}


# 89 - Confirm Output Files ----------------------------------------------------

output_files <- list.files(
  "outputs",
  full.names = TRUE
)


model_files <- list.files(
  "models",
  full.names = TRUE
)


cat(
  "\n============================================================\n"
)

cat(
  "OUTPUT FILES CREATED\n"
)

cat(
  "============================================================\n\n"
)


print(
  output_files
)


cat(
  "\n============================================================\n"
)

cat(
  "MODEL FILES CREATED\n"
)

cat(
  "============================================================\n\n"
)


print(
  model_files
)


# 90 - Session Information ----------------------------------------------------

sessionInfo()
