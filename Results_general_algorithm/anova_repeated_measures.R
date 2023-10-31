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
# No significant differences with accuracy_mean_mean

res.aov <- anova_test(
  data = datos,formula = accuracy_mean_median ~ weights*split*alpha,
  dv = accuracy_mean_mean, wid = Dataset,
  within = c(weights, split,alpha)
)
get_anova_table(res.aov, correction = 'GG')
# No significant differences with accuracy_mean_median

res.aov <- anova_test(
  data = datos,formula = accuracy_mean_std ~ weights*split*alpha,
  dv = accuracy_mean_mean, wid = Dataset,
  within = c(weights, split,alpha)
)
get_anova_table(res.aov, correction = 'GG')
# Significant differences: weights, split, alpha, weights:split


# pairwise comparisons
# P-values are adjusted using the Bonferroni multiple testing correction method.
pwc_split <- datos %>%
  pairwise_t_test(
    accuracy_mean_std ~ split, paired = TRUE,
    p.adjust.method = "bonferroni"
  )
pwc_split



datos %>%
  group_by(split) %>%
  summarise_at(vars(accuracy_mean_std),
               list(Mean_split = mean))
# the higher the value of split, the higher the variance


pwc_alpha <- datos %>%
  pairwise_t_test(
    accuracy_mean_std ~ alpha, paired = TRUE,
    p.adjust.method = "bonferroni"
  )
pwc_alpha

datos %>%
  group_by(alpha) %>%
  summarise_at(vars(accuracy_mean_std),
               list(Mean_alpha = mean))
# the higher the value of alpha, the higher the variance


# Pairwise comparisons: esta no tiene sentido porque la triple interacción es no significativa
pwc2 <- datos %>%
  group_by(alpha,weights) %>%
  pairwise_t_test(
    accuracy_mean_std ~ split, paired = TRUE,
    p.adjust.method = "bonferroni"
  )
pwc2

# tb puedo probar con un lmer
pwc <- selfesteem %>%
  pairwise_t_test(
    score ~ time, paired = TRUE,
    p.adjust.method = "bonferroni"
  )
pwc


################# LMER ####################

## Libraries
library(dplyr)
library(ggplot2)
library(lme4)
library(lmerTest)
library(tidyverse)
library(emmeans)
library(optimx)
library(ggsignif)


m1b = lmer(accuracy_mean_std ~ 1 + weights + split + alpha + 
             (1 + weights + split + alpha  | Dataset), datos,
           control=lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e5))) 
summary(m1b)


emm = emmeans(m1b, ~ split)
pairs(emm)

### Residual Checks
# https://www.rensvandeschoot.com/tutorials/lme4/
plot(fitted(m1b), resid(m1b, type = "pearson")) #
abline(0,0, col="red")

qqnorm(resid(m1b)) 
qqline(resid(m1b), col = "red") # add a perfect fit line

# random intercept subject
qqnorm(ranef(m1b)$Subject[,1] )
qqline(ranef(m1b)$Subject[,1], col = "red")
