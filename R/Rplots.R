#GGRPlots
library(ggplot2)
library(reshape2)

# PLOT COVERAGES
res01 <- CalculateCoverage("SIGMA_0.1", 0.1)
res03 <- CalculateCoverage("SIGMA_0.3", 0.3)
res05 <- CalculateCoverage("SIGMA_0.5", 0.5)
res07 <- CalculateCoverage("SIGMA_0.7", 0.7)
res1  <- CalculateCoverage("SIGMA_1", 1)

RES   <- rbind(res01, res03, res05, res07, res1)

RES_melt <- melt(RES)

#then plot
p <- ggplot(RES_melt, aes(x=factor(X1),y=value,fill=factor(X2)))+
 geom_boxplot() + labs(title="Coverage", y="Coverage",x="Sigma") 
p