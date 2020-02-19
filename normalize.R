
# Normalize dMRI features

df <- read.csv('~/pyProjects/CannnabisML/test/id_grp_sex_age_dmri28.csv', header=TRUE)

model_res <- c()
for (feat in 6:117){
  model <- glm(df[,feat] ~ df$Sex + df$Age + df$Site, family = "gaussian")
  model_res <- cbind(model_res, residuals(model))
}

model_res_df <- as.data.frame(matrix(model_res, nrow=length(df[,1]), byrow=FALSE))
write.table(model_res_df, "~/pyProjects/CannnabisML/test/res.dmri_03.csv", row.names=FALSE, col.names=FALSE, sep=",")



