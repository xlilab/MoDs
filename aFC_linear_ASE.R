source("~/script/system_config/load_Rpackage.R")
library(DescTools) #binomialCI
library(pracma) #eps
library(GenBinomApps)#clopper.pearson.ci 


##########################################################
###########Expression linear regression ##################
##########################################################

group_tissue <- fread("/picb/lilab5/dongdanyue/dongdanyue/Dosage/group_tissue.txt")
args <- commandArgs(T)
tissue <- args[1]
lowertissue <- group_tissue$lower[which(group_tissue$covariant_tissue == tissue)]
grouptissue <- group_tissue$tissue[which(group_tissue$covariant_tissue == tissue)]

allvar.df <- fread(paste("/picb/lilab5/dongdanyue/dongdanyue/dosage_sensitive/aFC/qtl/",tissue,"_qtl.txt",sep = ""))
cov <- fread(paste("/picb/lilab5/dongdanyue/dongdanyue/dosage_sensitive/aFC/covariants/",tissue,"_covariants.txt",sep = ""))
colnames(cov)[1] <-"CovID"
cov_melt <- melt(cov,id.vars = "CovID")
cov_melt <- dcast(cov_melt,variable~CovID)


TPM <- fread(paste("/picb/lilab5/datasets/restricted/gtex/v8/rna_seq/RNA_TPMs_by_tissue/gene_tpm_2017-06-05_v8_",lowertissue,".gct.gz",sep = ""))
sample_wgs <- fread("/picb/lilab5/dongdanyue/lilab5/gtex_20210826/data_file/all838sample_id.txt",header = F)
group <- fread("/picb/lilab5/dongdanyue/dongdanyue/GTEx_expression/group_pheno.txt")
group <- group %>% mutate(SUBID = str_extract(indID,"GTEX-[0-9,a-z,A-Z]*"))
tissue2 = gsub("- ","",group$group)
tissue2 = gsub(" ","_",tissue2)
tissue2 = gsub("\\(","",tissue2)
tissue2 = gsub("\\)","",tissue2)
group<-group %>% mutate(tissue = tissue2)%>% select(group,indID,SUBID,tissue) 
group <- group %>% filter(SUBID %in% sample_wgs$V1)
sample <- group %>% filter(tissue == grouptissue)
TPM_t.df <- TPM %>% select(Name,one_of(sample$indID)) 

alltissue.df <- data.frame()

chrall <- c(1:22,"X")
for(i in chrall){
  chr_tissue.df <- data.frame()
  print(i)
  var_chr <- allvar.df %>% filter(sid_chr == paste("chr",i,sep = ""))
  gt <- fread(paste("/picb/lilab5/dongdanyue/dongdanyue/dosage_sensitive/output/metasoft_eqtl_genotype/chr",i,"_genotype_simple.txt",sep = ""))
  gt <- gt %>% select(-c(CHROM,POS,FORMAT))
  for(j in 1:nrow(var_chr)){
    print(j)
    varID <- var_chr[j,1]
    varGene <- var_chr[j,2]
    gt_filter <- gt %>% dplyr::filter(ID == as.character(varID)) 
    gt_df <- melt(gt_filter,id.vars = c("ID")) #838
    gt_df <- gt_df %>% rename_dt(SUBID = variable,genotype = value)
    
    
    exp <- TPM_t.df %>% filter(Name == as.character(varGene))# new
    exp.df <- melt(exp,id.vars = "Name")
    exp.df <- exp.df%>% mutate(SUBID = str_extract(variable,"GTEX-[0-9,a-z,A-Z]*")) %>% 
      rename_dt(TPM = value)
    gt_df <- gt_df %>% inner_join(exp.df)
    gt_df <- gt_df %>%filter(!is.na(genotype))
    
    gt_df <- gt_df %>% left_join(cov_melt,by = c("SUBID" = "variable"))
    maf <- sum(gt_df$genotype)/(2*nrow(gt_df))
    formula <- paste("TPM ~ genotype+",paste(colnames(gt_df)[-c(1:6)],collapse = "+"))
    lm.m <- lm(formula = formula,gt_df)
    sm <-summary(lm.m)
    # slope <- sm$coefficients[2,]
    # names(slope) <- paste("slope",names(slope),sep ="_")
    # int <- sm$coefficients[1,]
    con<- confint(lm.m)
    lm.df <- cbind(sm$coefficients[1:2,],con[1:2,])
    lm.df <-data.frame(lm.df) 
    lm.df <- lm.df%>% mutate(variantID = as.character(varID),Gene =as.character(varGene),maf= maf,type = rownames(lm.df),tissue =tissue,adj.squar = sm$adj.r.squared,rquar = sm$r.squared) 
    lm.df <- lm.df %>% filter(type %in% c("(Intercept)","genotype"))
    chr_tissue.df <- rbind(chr_tissue.df,lm.df)
  }
  ####single chromosome linear regression result
  fwrite(chr_tissue.df,file = paste("/picb/lilab5/dongdanyue/dongdanyue/Dosage//linearout/",tissue,"_metasofteqtl_lineareQTL_chr",i,".txt",sep = ""),sep = "\t",quote = F,row.names = F,col.names = T,na = NA) 
  alltissue.df <- rbind(alltissue.df,chr_tissue.df)
}
###all chromosome results
fwrite(alltissue.df,file = paste("/picb/lilab5/dongdanyue/dongdanyue/Dosage//linearout/",tissue,"_metasofteqtl_lineareQTL_allchr.txt",sep = ""),sep = "\t",quote = F,row.names = F,col.names = T,na = NA) 




##########################################################
#########calculate aFC from linear regression ############
##########################################################
data_select <- alltissue.df %>% select(variantID, Gene,tissue, Estimate,X2.5..,X97.5..,type)
genotype <- data_select %>% filter(type == "genotype")
colnames(genotype)[4] <- "genotype"
genotype$type <- NULL
intercept <- data_select %>% filter(type == "(Intercept)")
intercept <- intercept %>% dplyr::select(variantID, Gene,tissue, Estimate)
colnames(intercept)[4] <- "intercept"
#####genotype contain CI 95% and estimate paramater
genotype <- genotype %>% left_join(intercept)
###if intercept less than zero,set it to minmun number that larger than zero
genotype <- genotype %>% mutate(intercept_correct = case_when(intercept>0 ~ intercept,
                                                              intercept<=0 ~ eps(1)))
###chose the lower bound as aFC
genotype <- genotype %>% mutate(genotypeCImin = case_when(X2.5.. * X97.5.. <=0 ~ 0,
                                                          X2.5..>0 & X97.5.. > 0 ~ X2.5..,
                                                          X2.5..<0 & X97.5.. < 0 ~ X97.5..))
genotype <- genotype %>% mutate(aFC_linear_min  = 2*genotypeCImin/intercept_correct +1)
genotype <- genotype %>% mutate(aFC_linear = 2*genotype/intercept_correct +1)

###Set a upper bound
genotype <- genotype %>% mutate(aFC_linear = case_when(aFC_linear < 0.01 ~ 0.01,
                                                       aFC_linear > 100 ~ 100,
                                                       aFC_linear >= 0.01 &  aFC_linear <=100 ~  aFC_linear))
genotype <- genotype %>% mutate(aFC_linear_min = case_when(aFC_linear_min < 0.01 ~ 0.01,
                                                           aFC_linear_min > 100 ~ 100,
                                                           aFC_linear_min >= 0.01 &  aFC_linear_min <=100 ~  aFC_linear_min))

out <- "/picb/lilab5/dongdanyue/dongdanyue/Dosage/ase_all_by_tissue/gt_ase_by_tissue/all_variant_linear_aFC.txt"
fwrite(genotype, file = out,sep = "\t",na = NA,quote = F,row.names = F,col.names = T)


##########################################################
#################calculate aFC from ASE###################
##########################################################

file = system("ls /picb/lilab5/dongdanyue/dongdanyue/Dosage/linearout/",intern = T) #all linear regression results
data <- fread(paste("/picb/lilab5/dongdanyue/dongdanyue/Dosage/linearout/",file,sep = ""))
tissue <- str_extract(file,".*(?=\\_metasoft)")
chr <-  str_extract(file,"(?<=lineareQTL\\_).*(?=\\.txt)")
input <- (paste("/picb/lilab5/dongdanyue/dongdanyue/Dosage//ase_all_by_tissue/gt_ase_by_tissue/",tissue,"/",chr,"_variant_linear_aFC.txt",sep = ""))
genotype <- fread(input)

### combine with ASE cacluated aFC
afc <- fread(paste("/picb/lilab5/dongdanyue/dongdanyue/Dosage//ase_all_by_tissue/gt_ase_by_tissue/",tissue,"/",chr,"_variant_ASE_aFC.txt",sep = ""))
afc_confit <- afc %>% filter(!is.na(median_ratio))
afc_confit <- afc_confit %>% rowwise() %>% mutate(BioCI = BinomCI(hap2_sum+1,hap2_sum+hap1_sum+2,method = "wald",conf.level = 0.95))
sumratio_CI <-  1/(afc_confit$BioCI) -1
colnames(sumratio_CI)  <- c("sumratio_estimate","sumratio_upper","sumratio_lower")
afc_confit <- cbind(afc_confit,sumratio_CI)
afc_confit$BioCI <- NULL

afc_confit <- afc_confit %>% mutate(ASE_median_min = case_when(lower < 1 & upper >1 ~ 1,
                                                               lower > 1 ~ lower,
                                                               upper < 1 ~ upper,
                                                               lower == 1 | upper == 1  ~ 1))
afc_confit <- afc_confit %>% mutate(ASE_sum_min = case_when(sumratio_lower < 1 & sumratio_upper >1 ~ 1,
                                                            sumratio_lower > 1 ~ sumratio_lower,
                                                            sumratio_upper < 1 ~ sumratio_upper,
                                                            sumratio_lower == 1 | sumratio_upper == 1  ~ 1))

afc_confit <- afc_confit %>% mutate(log2_ASE_median_min = log2(ASE_median_min),
                                    log2_ASE_sum_min = log2(ASE_sum_min))

afc_confit <- afc_confit %>% mutate(log2_ASE_median_min = case_when(log2_ASE_median_min < log2(0.01) ~ log2(0.01),
                                                                    log2_ASE_median_min > log2(100)  ~ log2(100) ,
                                                                    log2_ASE_median_min >=  log2(0.01)  &  log2_ASE_median_min <=log2(100) ~  log2_ASE_median_min))
###
afc_confit <- afc_confit %>% mutate(log2_ASE_sum_min = case_when(log2_ASE_sum_min < log2(0.01) ~ log2(0.01),
                                                                 log2_ASE_sum_min > log2(100)  ~ log2(100) ,
                                                                 log2_ASE_sum_min >=  log2(0.01)  &  log2_ASE_sum_min <=log2(100) ~  log2_ASE_sum_min))
afc_confit <- afc_confit %>% rowwise()%>% mutate(ASE_all_max_abslog2  = max(abs(log2_ASE_median_min),
                                                                            abs(log2_ASE_sum_min)))


#######save result
out <- paste("/picb/lilab5/dongdanyue/dongdanyue/Dosage/output/ASE_aFC/new/",tissue,"/",chr,"_variant_ASE_aFC_CI.txt",sep = "")
fwrite(afc_confit ,file = out,sep = "\t",na = NA,quote = F,row.names = F,col.names = T)

genotype <- genotype %>% mutate(log2_aFC_linear_min = log2(aFC_linear_min))
all <- genotype %>% left_join(afc_confit,by = c("variantID" = "variants_id" ,"Gene" = "gene_id","tissue" ))
all_min <- all %>% rowwise() %>%  mutate(max_aFC_3method = max(abs(log2_aFC_linear_min),ASE_all_max_abslog2,na.rm = T))

out <- paste("/picb/lilab5/dongdanyue/dongdanyue/Dosage/output/aFC/new/",tissue,"/",chr,"_all_maxCI.txt",sep = "")
fwrite(all_min ,file = out,sep = "\t",na = NA,quote = F,row.names = F,col.names = T)

