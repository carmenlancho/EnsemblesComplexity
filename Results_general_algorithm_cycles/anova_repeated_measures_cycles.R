library(tidyverse)
library(ggpubr)
library(rstatix)
library(datarium)

getwd()
setwd("/home/carmen/PycharmProjects/EnsemblesComplexity/Results_general_algorithm_cycles")
#datos <- read.csv('TotalAggregatedResults_ParameterConfiguration_CDB.csv') 
# Cargamos los datos ya agregados por medida de complejidad porque si no peta
datos <- read.csv('df_summary_data.csv') 
str(datos)
# Convert id and time into factor variables
datos <- datos %>%
  convert_as_factor(Dataset, combo_alpha_split, n_cycle,n_ensemble)

# Tenemos que hacer el análisis para cada combo_alpha_split
valores_combo = levels(datos$combo_alpha_split)
n_combo = length(valores_combo)
combo_friedman = data.frame(valores_combo)
combo_friedman$p_value = rep(NA,n_combo)

for (i in valores_combo){
  print(i)
  datos_i = datos[datos$combo_alpha_split==i,]
  fri = friedman.test(accuracy_mean_mean ~ n_cycle |Dataset, data=as.matrix(datos_i))
  combo_friedman[combo_friedman$valores_combo==i,2] = fri$p.value
}


res.aov <- anova_test(
  data = datos_a,formula = accuracy_mean_mean ~ n_cycle,
  dv = accuracy_mean_mean, wid = Dataset,
  within = c(n_cycle)
)
get_anova_table(res.aov, correction = 'GG')
# Significant differences with accuracy_mean_mean


# Normality check
plot(res.aov)
summary(res.aov)
res = attributes(res.aov)$args$model$residuals
index_sample <- sample(1:length(res),4000)
res_sample <- res[index_sample]

shapiro.test(as.numeric(res_sample))
hist(res)
# No cumple las hipótesis para versión paramétrica


# pairwise comparisons (versión paramétrica)
# P-values are adjusted using the Bonferroni multiple testing correction method.
pwc_split <- datos_a %>%
  pairwise_t_test(
    accuracy_mean_mean ~ n_cycle, paired = TRUE,
    p.adjust.method = "bonferroni"
  )
pwc_split


# Friedman, no paramétrico
res.fried <- datos_a %>% friedman_test(accuracy_mean_mean ~ n_cycle |Dataset)
res.fried$p

# no paramétrico, pairwise comparisons
# https://www.datanovia.com/en/lessons/friedman-test-in-r/
pwc2 <- datos_a %>% 
  wilcox_test(accuracy_mean_mean ~ n_cycle, paired = TRUE, p.adjust.method = "bonferroni")
pwc2

#Pairwise comparisons using sign test:
# Note that, it is also possible to perform pairwise comparisons using Sign Test, 
# which may lack power in detecting differences in paired data sets. However, 
# it is useful because it has few assumptions about the distributions of the data to compare
pwc2 <- datos_a %>%
  sign_test(accuracy_mean_mean ~ n_cycle, p.adjust.method = "bonferroni")
pwc2
# Mejor usamos el otro








datos %>%
  group_by(split) %>%
  summarise_at(vars(accuracy_mean_std),
               list(Mean_split = mean))
# the higher the value of split, the higher the variance




