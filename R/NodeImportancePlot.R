# Node Importance Plot
files <- list.files()
id    <- grep("feature_mask", files)
files <- files[id]

RES <- sapply(files,read.table)
RES <- do.call("rbind",RES)

IMP <- colMeans(RES)

# Get gene names
gene_names <- read.table("gene_names.txt")[[1]]

names(IMP) <- gene_names
IMP_s <- sort(IMP, decreasing=TRUE)

sel <- c("EGR2","HOXA2","HOXB13","HOXB2","MEIS1","MEIS2","TEAD1","TEAD4")
#barplot(IMP_s, horiz=TRUE, las=2)
barplot(sort(IMP_s[sel], decreasing=FALSE), horiz=TRUE, las=2, 
					col="cadetblue", xlab="Importance")

