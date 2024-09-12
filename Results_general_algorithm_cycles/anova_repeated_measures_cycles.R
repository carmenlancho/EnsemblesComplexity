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
combo_friedman[combo_friedman$p_value> 0.05]
# es decir, en todos los casos hay diferencias significativas

# Una vez hemos visto que existen diferencias significativas entre al menos un valor del combo
# hacemos las comparaciones múltiples

dif_no_sig <- data.frame(valores_combo)
dif_no_sig$niveles = rep(NA,n_combo)

for (i in valores_combo){
  print(i)
  datos_i = datos[datos$combo_alpha_split==i,]
  datos_i$n_cycle <- factor(datos_i$n_cycle) # los niveles del factor cambian en cada subset
  pwc2 <- datos_i %>% 
    wilcox_test(accuracy_mean_mean ~ n_cycle, paired = TRUE, p.adjust.method = "bonferroni")
  # Filtrar comparaciones con diferencias no significativas (suponiendo un umbral de p > 0.05)
  no_significativas <- pwc2[pwc2$p.adj>0.1,]

  
  # si no todas las comparaciones con ese nivel son no significativas, lo quitamos 
  # es decir, no nos vale que solo no haya diferencia entre 3 y 5 y con el resto (3-6,3-7,etc) sí
  max_cycles = max(as.numeric(pwc2$group2))
  valores_check <- unique(as.numeric(no_significativas$group1))
  for (v in valores_check){
    if (sum(no_significativas$group1 == v) <(max_cycles - v) ){
      no_significativas = no_significativas[no_significativas$group1!=v,]
    }
  }
  
  # Extraer los niveles de los pares con diferencias no significativas
  niveles_no_significativos <- unique(c(no_significativas$group1, no_significativas$group2))

  dif_no_sig[dif_no_sig$valores_combo==i,2] = paste(niveles_no_significativos, collapse = ", ")
}

write.csv(dif_no_sig, "CDB_cycles_ParametersComboAlphaSplit_dif_no_signif_cycles_mean.csv")

i = "alpha14-split28" 
datos_a = datos[datos$combo_alpha_split==i,]
datos_a$n_cycle <- factor(datos_a$n_cycle) # los niveles del factor cambian en cada subset

# no paramétrico, pairwise comparisons
# https://www.datanovia.com/en/lessons/friedman-test-in-r/
pwc2 <- datos_a %>% 
  wilcox_test(accuracy_mean_mean ~ n_cycle, paired = TRUE, p.adjust.method = "bonferroni")
pwc2

aa = pwc2[pwc2$p.adj>0.05,]

# Filtrar comparaciones con diferencias no significativas (suponiendo un umbral de p > 0.05)
no_significativas <- pwc2[pwc2$p.adj>0.05,]
max_cycles = max(as.numeric(pwc2$group2))
valores_check <- unique(as.numeric(no_significativas$group1))
for (v in valores_check){
  print(v)
  if (sum(no_significativas$group1 == v) <(max_cycles - v) ){ # si no todas las comparaciones con ese nivel son no significativas, lo quitamos
    no_significativas = no_significativas[no_significativas$group1!=v,]
  }
}


# Extraer los niveles de los pares con diferencias no significativas
niveles_no_significativos <- unique(c(no_significativas$group1, no_significativas$group2))
# Esto es un poco burdo porque pueden existir diferencias entre 3 y 5 y no entre 5 y 7, pero es demasiada info
# y lo que buscamos es el patrón



# res.aov <- anova_test(
#   data = datos_a,formula = accuracy_mean_mean ~ n_cycle,
#   dv = accuracy_mean_mean, wid = Dataset,
#   within = c(n_cycle)
# )
# get_anova_table(res.aov, correction = 'GG')
# # Significant differences with accuracy_mean_mean
# 
# 
# # Normality check
# plot(res.aov)
# summary(res.aov)
# res = attributes(res.aov)$args$model$residuals
# index_sample <- sample(1:length(res),4000)
# res_sample <- res[index_sample]
# 
# shapiro.test(as.numeric(res_sample))
# hist(res)
# No cumple las hipótesis para versión paramétrica


# pairwise comparisons (versión paramétrica)
# P-values are adjusted using the Bonferroni multiple testing correction method.
# pwc_split <- datos_a %>%
#   pairwise_t_test(
#     accuracy_mean_mean ~ n_cycle, paired = TRUE,
#     p.adjust.method = "bonferroni"
#   )
# pwc_split


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




