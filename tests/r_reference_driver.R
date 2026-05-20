#!/usr/bin/env Rscript
# Drive CRAN MatrixEQTL on its own bundled example dataset.
#
# Usage:
#   Rscript r_reference_driver.R <out_dir>
#
# Outputs (in out_dir), TSV with columns snps, gene, beta, statistic,
# pvalue, FDR -- one file per model x (cis | trans | all):
#   linear_all.tsv      modelLINEAR,        all-pairs (no cis/trans split)
#   linear_cis.tsv      modelLINEAR,        cis
#   linear_trans.tsv    modelLINEAR,        trans
#   anova_all.tsv       modelANOVA,         all-pairs
#   anova_cis.tsv       modelANOVA,         cis
#   anova_trans.tsv     modelANOVA,         trans
#   cross_all.tsv       modelLINEAR_CROSS,  all-pairs
#   cross_cis.tsv       modelLINEAR_CROSS,  cis
#   cross_trans.tsv     modelLINEAR_CROSS,  trans
#   info.tsv            dataset metadata + per-model counts

suppressPackageStartupMessages({
  library(MatrixEQTL)
})

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[[1]] else "R_out"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

base <- file.path(find.package("MatrixEQTL"), "data")

load_sd <- function(fname) {
  sd <- SlicedData$new()
  sd$fileDelimiter <- "\t"
  sd$fileOmitCharacters <- "NA"
  sd$fileSkipRows <- 1
  sd$fileSkipColumns <- 1
  sd$fileSliceSize <- 2000
  sd$LoadFile(file.path(base, fname))
  sd
}

snps <- load_sd("SNP.txt")
gene <- load_sd("GE.txt")
cvrt <- load_sd("Covariates.txt")

snpspos <- read.table(file.path(base, "snpsloc.txt"), header = TRUE,
                      stringsAsFactors = FALSE)
genepos <- read.table(file.path(base, "geneloc.txt"), header = TRUE,
                      stringsAsFactors = FALSE)

cisDist <- 1e6

write_eqtls <- function(df, path) {
  if (is.null(df) || nrow(df) == 0) {
    df <- data.frame(snps = character(), gene = character(),
                     beta = numeric(), statistic = numeric(),
                     pvalue = numeric(), FDR = numeric())
  }
  if (!("beta" %in% colnames(df))) df$beta <- NA_real_
  df <- df[, c("snps", "gene", "beta", "statistic", "pvalue", "FDR")]
  write.table(df, path, sep = "\t", quote = FALSE, row.names = FALSE)
}

models <- list(
  linear = modelLINEAR,
  anova  = modelANOVA,
  cross  = modelLINEAR_CROSS
)

info <- list()

for (mn in names(models)) {
  um <- models[[mn]]

  # --- all-pairs (trans-only, no cis/trans split) -------------------
  me_all <- Matrix_eQTL_engine(
    snps = snps$Clone(), gene = gene$Clone(), cvrt = cvrt$Clone(),
    output_file_name = NULL, pvOutputThreshold = 1,
    useModel = um, errorCovariance = numeric(),
    verbose = FALSE, pvalue.hist = FALSE)
  write_eqtls(me_all$all$eqtls, file.path(out_dir, paste0(mn, "_all.tsv")))

  # --- cis / trans split -------------------------------------------
  me_ct <- Matrix_eQTL_main(
    snps = snps$Clone(), gene = gene$Clone(), cvrt = cvrt$Clone(),
    output_file_name = NULL, pvOutputThreshold = 1,
    useModel = um, errorCovariance = numeric(), verbose = FALSE,
    output_file_name.cis = NULL, pvOutputThreshold.cis = 1,
    snpspos = snpspos, genepos = genepos,
    cisDist = cisDist, pvalue.hist = FALSE,
    min.pv.by.genesnp = FALSE, noFDRsaveMemory = FALSE)
  write_eqtls(me_ct$cis$eqtls, file.path(out_dir, paste0(mn, "_cis.tsv")))
  write_eqtls(me_ct$trans$eqtls, file.path(out_dir, paste0(mn, "_trans.tsv")))

  info[[length(info) + 1]] <- data.frame(
    model = mn,
    n_all = nrow(me_all$all$eqtls),
    n_cis = me_ct$cis$neqtls,
    n_trans = me_ct$trans$neqtls,
    ntests_cis = me_ct$cis$ntests,
    ntests_trans = me_ct$trans$ntests
  )
}

info_df <- do.call(rbind, info)
write.table(info_df, file.path(out_dir, "info.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

cat("R MatrixEQTL reference done\n")
