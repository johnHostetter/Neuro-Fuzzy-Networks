library(RoughSets)
###########################################################
## Example 1: Regression problem
###########################################################
# data(RoughSetData)
# decision.table <- RoughSetData$housing7.dt
# control <- list(type.aggregation = c("t.tnorm", "lukasiewicz"), type.relation =
#                   c("tolerance", "eq.3"), t.implicator = "lukasiewicz")
# res.1 <- RI.hybridFS.FRST(decision.table, control)
# summary(res.1)
###########################################################
## Example 2: Classification problem
##############################################################
# data(RoughSetData)
# decision.table <- RoughSetData$pima7.dt
# control <- list(type.aggregation = c("t.tnorm", "lukasiewicz"), type.relation =
#                   c("tolerance", "eq.3"), t.implicator = "lukasiewicz")
# res.2 <- RI.hybridFS.FRST(decision.table, control)
# summary(res.2)

setwd("~/Documents/ncsu2021/FNN/RSPOP")
my_data <- read.csv("test.csv")
decision.table <- SF.asDecisionTable(head(my_data, 10), decision.attr = NULL, indx.nominal = NULL)
control <- list(type.aggregation = c("t.tnorm", "lukasiewicz"), type.relation =
                  c("tolerance", "eq.3"), t.implicator = "lukasiewicz")
res.3 <- RI.hybridFS.FRST(decision.table, control)
summary(res.3)

