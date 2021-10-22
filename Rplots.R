# R plots
# For our modified GNNexplainer #############################
folder <- "graphs_3_11"
path   <- paste(folder,"/dataset/graph0_edges.txt", sep="")
edges  <- read.table(path)

edges  <- apply(edges,2,function(x){paste(x[1],x[2],sep="-")})

path2  <- paste("./",folder,"/0.1/modified_gnn/", sep="")
files  <- list.files(path2, full.names = TRUE)

RES <- vector("list", length(files))
for(xx in 1:length(files)){

	RES[[xx]] <- read.csv(files[xx], header=FALSE)

}

SCORES <- t(do.call("cbind",RES))
#boxplot(SCORES, names=edges)
boxplot(SCORES, outline=FALSE, boxwex=0.4, 
	color="white", names=edges, las=2)


RANK <- t(apply(SCORES, 1, function(x){rank(x)}))
boxplot(RANK, outline=FALSE, boxwex=0.4, 
	color="white", names=edges, las=2)

##############################################################
# For the PGExplainer ########################################
folder <- "graphs_2_19"
path   <- paste(folder,"/dataset/graph0_edges.txt", sep="")
edges  <- read.table(path)

edges  <- apply(edges,2,function(x){paste(x[1],x[2],sep="-")})

path2  <- paste("./",folder,"/pg_results/", sep="")
files  <- list.files(path2, full.names = TRUE)

RES <- vector("list", length(files))
for(xx in 1:length(files)){

	RES[[xx]] <- read.csv(files[xx], header=FALSE)

}
SCORES <- abs(RES[[1]]) #t(do.call("cbind",RES))
#boxplot(SCORES, names=edges)
range01        <- function(x){(x-min(x))/(max(x)-min(x))}
SCORES2 <- t(apply(SCORES,1,range01))
boxplot(SCORES2, outline=FALSE, boxwex=0.4, 
	color="white", names=edges, las=2, cex.axis=0.6)

RANK <- t(apply(SCORES, 1, function(x){rank(x)}))
boxplot(RANK, outline=FALSE, boxwex=0.4, 
	color="white", names=edges, las=2)