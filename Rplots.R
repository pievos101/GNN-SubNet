PG_CalcEdgeImportance <- function(EdgeScores, Edges){


EDGE_SCORES <- rep(NaN,dim(Edges)[2])

for(xx in 1:dim(Edges)[2]){
	
node1 <- Edges[1,xx]
node2 <- Edges[2,xx]

ids1  <- which((Edges[1,]==node1) | (Edges[1,]==node2) )
ids2  <- which((Edges[2,]==node1) | (Edges[2,]==node2) )

ES <- EdgeScores[,unique(c(ids1,ids2))]
EDGE_SCORES[xx] <- mean(apply(ES,2,mean))

}

return(EDGE_SCORES)
}

##############################################################
# For the PGExplainer ########################################
folder <- "graphs_3_7"
path   <- paste(folder,"/dataset/graph0_edges.txt", sep="")
edges_raw  <- read.table(path)

edges  <- apply(edges_raw,2,function(x){paste(x[1],x[2],sep="-")})

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


newScores <- PG_CalcEdgeImportance(SCORES2, edges_raw)
names(newScores) <- edges

barplot(sort(newScores, decreasing=TRUE), las=2, cex.names=0.6)

#RANK <- t(apply(SCORES, 1, function(x){rank(x)}))
#boxplot(RANK, outline=FALSE, boxwex=0.4, 
#	color="white", names=edges, las=2)






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