library(gtools)
library(tidyverse)
medals<-c('g','s','b')
runners <- 1:8
permutations(8,3,c(1,2,3,4,5,6,7,8))

permutations(3,3,c(1,2,3))

entrees <- paste('e', 1:6)
drink <- c('d1','d2', 'd3')
sides <- paste('s', 1:6)

sides_comb <- combinations(6,3,v=sides)
no_sides <- expand.grid(entrees, drink)
nrow(no_sides) * nrow(sides_comb)

head(esoph)
length(esoph$agegp)
all_cases <- sum(esoph$ncases)
all_controls <- sum(esoph$ncontrols)
summary(esoph)

algp120 <- esoph %>% filter(alcgp == '120+')
sum(algp120$ncases)/(sum(algp120$ncases)+sum(algp120$ncontrols))

algplow <- esoph %>% filter(alcgp == '0-39g/day')
sum(algplow$ncases)/(sum(algplow$ncases)+sum(algplow$ncontrols))

sum(algp120$ncases)/all_cases

tbgp30 <- esoph %>% filter(tobgp == '30+')
algp120tbggp30 <- esoph %>% filter(alcgp == '120+' & tobgp == '30+')
algp120tbggp30or <- esoph %>% filter(alcgp == '120+' | tobgp == '30+')
sum(algp120tbggp30$ncases)/all_cases
sum(algp120tbggp30or$ncases)/all_cases

(sum(algp120$ncases)/all_cases)/(sum(algp120$ncontrols)/all_controls)
sum(tbgp30$ncontrols)/all_controls