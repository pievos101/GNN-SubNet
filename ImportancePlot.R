ImportancePlot <- function(folder){

# R plots
# For our modified GNNexplainer #############################
#folder <- "graphs_5_18"
path       <- paste(folder,"/edge_index.txt", sep="")
edges_raw  <- read.table(path)

edges  <- apply(edges_raw,2,function(x){paste(x[1],x[2],sep="-")})

path2  <- paste(folder,"", sep="")
files  <- list.files(path2, full.names = TRUE)


#####################
# Node Importance
#####################

par(mfrow=c(2,1))
filesX  <- files[grep("gnn_feature_mask", files)]
RES <- vector("list", length(filesX))
for(xx in 1:length(filesX)){

	RES[[xx]] <- read.csv(filesX[xx], header=FALSE)

}

SCORES <- t(do.call("cbind",RES))
#boxplot(SCORES, names=edges)
boxplot(SCORES, outline=FALSE, boxwex=0.4, 
	color="white", names=0:(dim(SCORES)[2]-1), las=2, xlab="Nodes", 
	ylab="Node Importance", col="cadetblue")

#####################
# Edge Importance
#####################

filesX  <- files[grep("gnn_edge_mask", files)]
RES <- vector("list", length(filesX))
for(xx in 1:length(filesX)){

	RES[[xx]] <- read.csv(filesX[xx], header=FALSE)

}

SCORES <- t(do.call("cbind",RES))
#boxplot(SCORES, names=edges)
boxplot(SCORES, outline=FALSE, boxwex=0.4, 
	color="white", names=edges, las=2, xlab="Edges", 
	ylab="Edge Importance", col="coral3")
}

ImportancePlotSim <- function(folder){

# R plots
# For our modified GNNexplainer #############################
#folder <- "graphs_5_18"
path   <- paste(folder,"/dataset/graph0_edges.txt", sep="")
edges_raw  <- read.table(path)

edges  <- apply(edges_raw,2,function(x){paste(x[1],x[2],sep="-")})

path2  <- paste("./",folder,"/0.1/modified_gnn/", sep="")
files  <- list.files(path2, full.names = TRUE)


#####################
# Node Importance
#####################
par(mfrow=c(2,1))
filesX  <- files[grep("feature_mask", files)]
RES <- vector("list", length(filesX))
for(xx in 1:length(filesX)){

	RES[[xx]] <- read.csv(filesX[xx], header=FALSE)

}

SCORES <- t(do.call("cbind",RES))
#boxplot(SCORES, names=edges)
boxplot(SCORES, outline=FALSE, boxwex=0.4, 
	color="white", names=0:(dim(SCORES)[2]-1), las=2, xlab="Nodes", 
	ylab="Node Importance", col="cadetblue")

colnames(SCORES) <- 0:(dim(SCORES)[2]-1)
NODE_IMP <- SCORES 

#####################
# Edge Importance
#####################
filesX  <- files[grep("edge_mask", files)]
RES <- vector("list", length(filesX))
for(xx in 1:length(filesX)){

	RES[[xx]] <- read.csv(filesX[xx], header=FALSE)

}

SCORES <- t(do.call("cbind",RES))
#boxplot(SCORES, names=edges)
boxplot(SCORES, outline=FALSE, boxwex=0.4, 
	color="white", names=edges, las=2, xlab="Edges", 
	ylab="Edge Importance", col="coral3")

colnames(SCORES) <- edges
EDGE_IMP <- SCORES 

return(list(NODE_IMP=NODE_IMP, EDGE_IMP=EDGE_IMP))

}