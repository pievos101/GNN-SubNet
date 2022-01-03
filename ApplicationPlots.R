# R Application plots
#####################################

comms <- read.table("~/LinkedOmics/KIRC-RANDOM/communities.txt")[[1]]
comms_scores <- read.table("~/LinkedOmics/KIRC-RANDOM/communities_scores.txt")[[1]]
genes <- read.table("~/LinkedOmics/KIRC-RANDOM/gene_names.txt")
genes <- unlist(genes)

id <- which.max(comms_scores)
comms_scores[id]
best_com <- comms[id]

best_com <- as.numeric(strsplit(best_com,",")[[1]])

# Best community
TOPCOMM  <- genes[best_com] 

edge_mask  <- read.table("~/LinkedOmics/KIRC-RANDOM/edge_masks.txt")[[1]]
edge_index <- read.table("~/LinkedOmics/KIRC-RANDOM/edge_index.txt")

ids1 <- is.element(edge_index[1,], best_com)
ids2 <- is.element(edge_index[2,], best_com)
IDS  <- ids1 & ids2

TOPCOMM_edges <- edge_index[,IDS] 

TOPCOMM_edges_names <- rbind(genes[as.numeric(TOPCOMM_edges[1,])], 
							genes[as.numeric(TOPCOMM_edges[2,])])
colnames(TOPCOMM_edges_names) <- NULL
TOPCOMM_edges_names 

edge_mask[IDS]