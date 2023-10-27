library(tidyverse)
library(ggpubr)
library(rstatix)
library(datarium)

getwd()
setwd("/home/carmen/PycharmProjects/EnsemblesComplexity/Results_general_algorithm")
datos <- read.csv('SummarizeResults_ParameterConfiguration_CDB.csv')
str(datos)
# Convert id and time into factor variables
datos <- datos %>%
  convert_as_factor(Dataset, alpha,split, weights)

# Normality assumption
# Compute Shapiro-Wilk test for each combinations of factor levels:
datos %>%
  group_by(alpha, split) %>%
  shapiro_test(accuracy_mean_std)

# Create QQ plot for each cell of design:
ggqqplot(datos, "accuracy_mean_std", ggtheme = theme_bw()) +
  facet_grid(alpha ~ split, labeller = "label_both")

res.aov <- anova_test(
  data = datos,formula = accuracy_mean_mean ~ weights*split*alpha,
  dv = accuracy_mean_mean, wid = Dataset,
  within = c(weights, split,alpha)
)
get_anova_table(res.aov, correction = 'GG')


set.seed(123)
data("selfesteem2", package = "datarium")
selfesteem2 %>% sample_n_by(treatment, size = 1)

# Gather the columns t1, t2 and t3 into long format.
# Convert id and time into factor variables
selfesteem2 <- selfesteem2 %>%
  gather(key = "time", value = "score", t1, t2, t3) %>%
  convert_as_factor(id, time)
# Inspect some random rows of the data by groups
set.seed(123)
selfesteem2 %>% sample_n_by(treatment, time, size = 1)

bxp <- ggboxplot(
  selfesteem2, x = "time", y = "score",
  color = "treatment", palette = "jco"
)
bxp

# Identify outliers
selfesteem2 %>%
  group_by(treatment, time) %>%
  identify_outliers(score)


# Normality assumption
# Compute Shapiro-Wilk test for each combinations of factor levels:

selfesteem2 %>%
  group_by(treatment, time) %>%
  shapiro_test(score)


# Create QQ plot for each cell of design:

ggqqplot(selfesteem2, "score", ggtheme = theme_bw()) +
  facet_grid(time ~ treatment, labeller = "label_both")


res.aov <- anova_test(
  data = selfesteem2, dv = score, wid = id,
  within = c(treatment, time)
)
get_anova_table(res.aov)
