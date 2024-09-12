CDB_cycles_AnalysisOfParameters
================

In this notebook, we are studying the different parameters of our method
Complexity Driven Bagging so as to offer a range of selection to the
final user. In particular, we have three parameters:

-   Split: the number of splits in which we cut the complexity spectrum.
    s=1 implies we are training with one easy sample, one uniform sample
    and one hard sample. That is, the cycle length is 3. s=2 implies 6
    samples of different complexity (cycle length is 6).

-   Alpha: to give more weight to the easiest and the hardest instances
    in the bootstrap sampling procedure with the aim of training the
    classifier with samples of higher or lower complexity (thus,
    enlarging the original range of complexity).

-   Number of cycles. How many times the procedure is repeated. This is
    totally related with the final number of ensembles.

Besides this 3 parameters, we have obtained results for different
complexity measures. For the analysis of the parameters, we have
aggregated results over the different complexity measures.

First, we are studying the recommended number of cycles. That is, given
a value of split and a value of alpha, we want to know when the best
accuracy is obtained, when there are significant differences, etc. so as
to recommend the lower number of cycles (lower number of ensembles) with
the best performance.

Once the number of cycles is reduced, we perform a similar analysis on
the alpha parameter and the on the split parameter.

# First view

``` r
library(tidyverse)
```

    ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.2 ──
    ✔ ggplot2 3.5.1     ✔ purrr   1.0.2
    ✔ tibble  3.2.1     ✔ dplyr   1.1.4
    ✔ tidyr   1.3.1     ✔ stringr 1.5.1
    ✔ readr   2.1.2     ✔ forcats 0.5.2
    ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ✖ dplyr::filter() masks stats::filter()
    ✖ dplyr::lag()    masks stats::lag()

``` r
library(ggpubr)
library(rstatix)
```


    Attaching package: 'rstatix'

    The following object is masked from 'package:stats':

        filter

``` r
library(datarium)

getwd()
```

    [1] "/home/carmen/PycharmProjects/EnsemblesComplexity/Results_general_algorithm_cycles"

``` r
setwd("/home/carmen/PycharmProjects/EnsemblesComplexity/Results_general_algorithm_cycles")
#datos <- read.csv('TotalAggregatedResults_ParameterConfiguration_CDB.csv') 
# Data aggregated over complexity measures
datos <- read.csv('df_summary_data.csv') 
str(datos)
```

    'data.frame':   156040 obs. of  9 variables:
     $ Dataset             : chr  "WineQualityRed_5vs6" "WineQualityRed_5vs6" "WineQualityRed_5vs6" "WineQualityRed_5vs6" ...
     $ n_cycle             : int  1 1 1 1 1 1 1 1 1 1 ...
     $ n_ensemble          : int  2 2 2 2 2 2 2 2 2 2 ...
     $ alpha               : int  2 4 6 8 10 12 14 16 18 20 ...
     $ split               : int  1 1 1 1 1 1 1 1 1 1 ...
     $ accuracy_mean_mean  : num  0.731 0.73 0.735 0.733 0.729 ...
     $ accuracy_mean_median: num  0.728 0.726 0.731 0.735 0.735 ...
     $ accuracy_mean_std   : num  0.00857 0.01481 0.01139 0.01202 0.01402 ...
     $ combo_alpha_split   : chr  "alpha2-split1" "alpha4-split1" "alpha6-split1" "alpha8-split1" ...

``` r
# Convert id and time into factor variables
datos <- datos %>%
  convert_as_factor(Dataset, combo_alpha_split, n_cycle,n_ensemble)
```

``` r
datos %>%
  group_by(split) %>%
  summarise_at(vars(accuracy_mean_mean),
               funs(mean,median, sd))
```

    Warning: `funs()` was deprecated in dplyr 0.8.0.
    ℹ Please use a list of either functions or lambdas:

    # Simple named list: list(mean = mean, median = median)

    # Auto named with `tibble::lst()`: tibble::lst(mean, median)

    # Using lambdas list(~ mean(., trim = .2), ~ median(., na.rm = TRUE))

    # A tibble: 16 × 4
       split  mean median    sd
       <int> <dbl>  <dbl> <dbl>
     1     1 0.811  0.797 0.121
     2     2 0.813  0.796 0.121
     3     4 0.813  0.795 0.121
     4     6 0.813  0.794 0.121
     5     8 0.813  0.794 0.121
     6    10 0.813  0.794 0.121
     7    12 0.814  0.794 0.121
     8    14 0.814  0.794 0.121
     9    16 0.814  0.794 0.121
    10    18 0.814  0.794 0.121
    11    20 0.814  0.794 0.121
    12    22 0.814  0.794 0.121
    13    24 0.814  0.794 0.121
    14    26 0.814  0.794 0.121
    15    28 0.814  0.795 0.121
    16    30 0.814  0.794 0.121

``` r
datos %>%
  group_by(alpha) %>%
  summarise_at(vars(accuracy_mean_mean),
               funs(mean,median, sd))
```

    Warning: `funs()` was deprecated in dplyr 0.8.0.
    ℹ Please use a list of either functions or lambdas:

    # Simple named list: list(mean = mean, median = median)

    # Auto named with `tibble::lst()`: tibble::lst(mean, median)

    # Using lambdas list(~ mean(., trim = .2), ~ median(., na.rm = TRUE))

    # A tibble: 10 × 4
       alpha  mean median    sd
       <int> <dbl>  <dbl> <dbl>
     1     2 0.813  0.796 0.121
     2     4 0.813  0.796 0.121
     3     6 0.813  0.796 0.121
     4     8 0.813  0.796 0.121
     5    10 0.813  0.796 0.121
     6    12 0.812  0.796 0.121
     7    14 0.812  0.796 0.121
     8    16 0.812  0.795 0.121
     9    18 0.812  0.795 0.121
    10    20 0.812  0.795 0.121

We cannot perform a summary of ‘n_cycle’ in general because the number
of cycles depends on the value of split. Thus, we show some cases.

split = 1

``` r
datos %>% filter(split == 1) %>%
  group_by(n_cycle) %>%
  summarise_at(vars(accuracy_mean_mean),
               funs(mean,median, sd))
```

    Warning: `funs()` was deprecated in dplyr 0.8.0.
    ℹ Please use a list of either functions or lambdas:

    # Simple named list: list(mean = mean, median = median)

    # Auto named with `tibble::lst()`: tibble::lst(mean, median)

    # Using lambdas list(~ mean(., trim = .2), ~ median(., na.rm = TRUE))

    # A tibble: 100 × 4
       n_cycle  mean median    sd
       <fct>   <dbl>  <dbl> <dbl>
     1 1       0.776  0.759 0.134
     2 2       0.786  0.774 0.130
     3 3       0.797  0.786 0.127
     4 4       0.799  0.788 0.126
     5 5       0.803  0.790 0.125
     6 6       0.804  0.791 0.124
     7 7       0.806  0.792 0.123
     8 8       0.807  0.793 0.123
     9 9       0.808  0.793 0.123
    10 10      0.808  0.793 0.123
    # ℹ 90 more rows

split = 2

``` r
datos %>% filter(split == 2) %>%
  group_by(n_cycle) %>%
  summarise_at(vars(accuracy_mean_mean),
               funs(mean,median, sd))
```

    Warning: `funs()` was deprecated in dplyr 0.8.0.
    ℹ Please use a list of either functions or lambdas:

    # Simple named list: list(mean = mean, median = median)

    # Auto named with `tibble::lst()`: tibble::lst(mean, median)

    # Using lambdas list(~ mean(., trim = .2), ~ median(., na.rm = TRUE))

    # A tibble: 60 × 4
       n_cycle  mean median    sd
       <fct>   <dbl>  <dbl> <dbl>
     1 1       0.788  0.776 0.130
     2 2       0.798  0.783 0.127
     3 3       0.804  0.788 0.124
     4 4       0.806  0.790 0.124
     5 5       0.808  0.791 0.123
     6 6       0.809  0.791 0.123
     7 7       0.810  0.792 0.122
     8 8       0.811  0.793 0.122
     9 9       0.811  0.793 0.122
    10 10      0.811  0.794 0.122
    # ℹ 50 more rows

split = 4

``` r
datos %>% filter(split == 4) %>%
  group_by(n_cycle) %>%
  summarise_at(vars(accuracy_mean_mean),
               funs(mean,median, sd))
```

    Warning: `funs()` was deprecated in dplyr 0.8.0.
    ℹ Please use a list of either functions or lambdas:

    # Simple named list: list(mean = mean, median = median)

    # Auto named with `tibble::lst()`: tibble::lst(mean, median)

    # Using lambdas list(~ mean(., trim = .2), ~ median(., na.rm = TRUE))

    # A tibble: 34 × 4
       n_cycle  mean median    sd
       <fct>   <dbl>  <dbl> <dbl>
     1 1       0.799  0.781 0.127
     2 2       0.805  0.787 0.125
     3 3       0.809  0.790 0.123
     4 4       0.810  0.792 0.123
     5 5       0.811  0.792 0.122
     6 6       0.812  0.793 0.122
     7 7       0.812  0.794 0.122
     8 8       0.813  0.793 0.122
     9 9       0.813  0.794 0.121
    10 10      0.813  0.794 0.121
    # ℹ 24 more rows

split = 10

``` r
datos %>% filter(split == 10) %>%
  group_by(n_cycle) %>%
  summarise_at(vars(accuracy_mean_mean),
               funs(mean,median, sd))
```

    Warning: `funs()` was deprecated in dplyr 0.8.0.
    ℹ Please use a list of either functions or lambdas:

    # Simple named list: list(mean = mean, median = median)

    # Auto named with `tibble::lst()`: tibble::lst(mean, median)

    # Using lambdas list(~ mean(., trim = .2), ~ median(., na.rm = TRUE))

    # A tibble: 15 × 4
       n_cycle  mean median    sd
       <fct>   <dbl>  <dbl> <dbl>
     1 1       0.807  0.788 0.124
     2 2       0.811  0.792 0.122
     3 3       0.812  0.794 0.122
     4 4       0.813  0.794 0.121
     5 5       0.814  0.795 0.121
     6 6       0.814  0.795 0.121
     7 7       0.814  0.795 0.121
     8 8       0.814  0.796 0.121
     9 9       0.814  0.795 0.121
    10 10      0.815  0.796 0.121
    11 11      0.815  0.795 0.121
    12 12      0.815  0.794 0.121
    13 13      0.815  0.795 0.121
    14 14      0.815  0.795 0.121
    15 15      0.815  0.796 0.121

# Number of cycles
