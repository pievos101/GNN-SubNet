#PPI    <- read.table("~/LinkedOmics/KIRC-RANDOM/KIDNEY_RANDOM_PPI.txt")
#genes  <- c("EGR2","HOXA2","HOXB13","HOXB2","MEIS1","MEIS2","TEAD1","TEAD4")
#ids    <- apply(PPI, 1, function(x){is.element(x[1],genes)&is.element(x[2],genes)})
#module <- PPI[which(ids),][PPI[which(ids),3]>=950,]

comms <- read.table("~/LinkedOmics/KIRC-RANDOM/communities.txt")[[1]]
comms_scores <- read.table("~/LinkedOmics/KIRC-RANDOM/communities_scores.txt")[[1]]
genes <- read.table("~/LinkedOmics/KIRC-RANDOM/gene_names.txt")
genes <- unlist(genes)

id <- which.max(comms_scores)
comms_scores[id]

best_com <- comms[id]

best_com <- as.numeric(strsplit(best_com,",")[[1]]) + 1

# Best community
TOPCOMM  <- genes[best_com] 


genes <- TOPCOMM

# Get importances
edge_imp <- read.table("~/LinkedOmics/KIRC-RANDOM/edge_masks.txt")
edge_imp <- edge_imp[[1]]

# Get edge ids
edges <- read.table("~/LinkedOmics/KIRC-RANDOM/edge_index.txt")
edges <- edges

colnames(edges) <- edge_imp

# Get genes
gene_names <- read.table("~/LinkedOmics/KIRC-RANDOM/gene_names.txt")
gene_names <- gene_names[[1]]


ii <- match(genes, gene_names)
ii <- ii - 1 #python starts with 1

ids <- apply(edges, 2, function(x){is.element(x[1],ii)&is.element(x[2],ii)})
ix  <- which(ids)

edges_module       <- t(edges[,ix])
edges_module_genes <- cbind(gene_names[edges_module[,1]+1],
							gene_names[edges_module[,2]+1])  
IMP <- edge_imp[ix]

g <- graph_from_edgelist(edges_module_genes, directed=TRUE)

plot(g, vertex.shape="none", vertex.color="black",layout=layout.circle, 
	edge.width = round(IMP*5, digits=3),
	edge.label = round(IMP, digits=3),
	edge.label.cex = 0.8)


