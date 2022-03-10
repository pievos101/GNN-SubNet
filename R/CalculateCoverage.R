source("~/GitHub/GNN-SubNet/ImportancePlot.R")

CalculateCoverage <- function(folder, sigma=0.7, topK=c(1,2,3,5)){

files <- list.files(folder, full.names=TRUE)
selected_edges <- strsplit(files,"graphs_")
selected_edges <- sapply(selected_edges,function(x){x[2]})	
#selected_edges <- strsplit(selected_edges,"_")
#selected_edges <- sapply(selected_edges,function(x){as.numeric(x)})	
selected_edges <- gsub("_", "-", selected_edges)

topK <- topK
#files <- files[1]
SIGMA_RES <- NULL
for (xx in 1:length(files)){

print(files[xx])
res <- ImportancePlotSim(files[xx], sigma=sigma) #@TODO	
EDGE_IMP <- res$EDGE_IMP
NODE_IMP <- res$NODE_IMP

eee        <- colnames(EDGE_IMP)
sel_edg_id <- selected_edges[xx]

# id is the selected edge within EDGE_IMP
id <- match(sel_edg_id, eee)

RANK <- apply(EDGE_IMP,1,function(x){rank(-x)})

	COV  <- numeric(length(topK))
	for(yy in 1:length(topK)){
		COV[yy] <- sum(RANK[id,]<=topK[yy])/length(RANK[id,])
	}

SIGMA_RES <- cbind(SIGMA_RES,COV)

}

SIGMA_RES <- t(SIGMA_RES)
colnames(SIGMA_RES) <- paste("top", topK, sep="")
rownames(SIGMA_RES) <- rep(paste("SIGMA_",sigma,sep=""),dim(SIGMA_RES)[1])
return(SIGMA_RES)
}