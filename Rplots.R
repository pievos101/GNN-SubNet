# R
# For our modified GNNexplainer
folder <- "graphs_10_16"
path   <- paste(folder,"/dataset/graph0_edges.txt", sep="")
edges  <- read.table(path)

edges  <- apply(edges,2,function(x){paste(x[1],x[2],sep="-")})

path2  <- paste("./",folder,"/0.5/modified_gnn/", sep="")
files  <- list.files(path2, full.names = TRUE)

RES <- vector("list", length(files))
for(xx in 1:length(files)){

	RES[[xx]] <- read.csv(files[xx])

}

SCORES <- t(do.call("cbind",RES))
boxplot(SCORES, names=edges)