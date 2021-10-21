# R plots
# For our modified GNNexplainer
folder <- "graphs_7_8"
path   <- paste(folder,"/dataset/graph0_edges.txt", sep="")
edges  <- read.table(path)

edges  <- apply(edges,2,function(x){paste(x[1],x[2],sep="-")})

path2  <- paste("./",folder,"/0.5/modified_gnn/", sep="")
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

# For the PGExplainer
