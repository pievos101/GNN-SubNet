# tmp

sigma = 0.7

files <- list.files(full.names=TRUE)
selected_edges <- strsplit(files,"graphs_")
selected_edges <- sapply(selected_edges,function(x){x[2]})	
#selected_edges <- strsplit(selected_edges,"_")
#selected_edges <- sapply(selected_edges,function(x){as.numeric(x)})	
selected_edges <- gsub("_", "-", selected_edges)

selected_edges <- strsplit(selected_edges,"-")

HIT   = rep(NaN, length(files))

for(xx in 1:length(files)){

 edge_index <- read.table(paste(files[xx],"/dataset/graph0_edges.txt", sep=""))
 edge_index <- edge_index + 1

 LOC = paste(files[xx],"/",sigma,"/vanilla_gnn/gnn_feature_masks1.csv", sep="")
 feat_imp = read.table(LOC)
 feat_imp = unlist(feat_imp)
 edge_imp = feat_imp[unlist(edge_index[1,])] + feat_imp[unlist(edge_index[2,])]
 sel = as.numeric(selected_edges[[xx]]) + 1
 #print(sel)
 id  = apply(edge_index,2,function(x){all(x==sel)})
 id  = which(id)
 #print(id)
 HIT[xx] = which.max(edge_imp) == id
 #print(hit)

}

sum(HIT)/length(HIT)