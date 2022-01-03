# R Application plots
#####################################

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

edge_mask  <- read.table("~/LinkedOmics/KIRC-RANDOM/edge_masks.txt")[[1]]
edge_index <- read.table("~/LinkedOmics/KIRC-RANDOM/edge_index.txt")

ids1 <- is.element(edge_index[1,]+1, best_com)
ids2 <- is.element(edge_index[2,]+1, best_com)
IDS  <- ids1 & ids2

TOPCOMM_edges <- edge_index[,IDS] 

TOPCOMM_edges_names <- rbind(genes[as.numeric(TOPCOMM_edges[1,]+1)], 
							genes[as.numeric(TOPCOMM_edges[2,]+1)])
colnames(TOPCOMM_edges_names) <- NULL
TOPCOMM_edges_names 

TOPCOMM_edge_imp = edge_mask[IDS]

MODULE <- cbind(t(TOPCOMM_edges_names), TOPCOMM_edge_imp)
colnames(MODULE) <- c("gene1","gene2","IMP")

# Plot the detected disease module
g  <- graph_from_edgelist(as.matrix(MODULE[,1:2]), directed=TRUE)

plot(g, vertex.shape="none", vertex.color="black",layout=layout.circle, 
	edge.width = round(as.numeric(MODULE[,3])*5, digits=3),
	edge.label = round(as.numeric(MODULE[,3]), digits=3),
	edge.label.cex = 0.8)
