#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""
Perform GWAS using OLS.
"""

import sys

import numpy as np
import pandas as pd
# from sklearn.prepro
import statsmodels.api as sm


# Pass file paths as arguments to this script and load into dataframes.
# gwas.py [genotype.csv] [FT10.txt]
genotype_file_path = sys.argv[1]
phenotype_file_path = sys.argv[2]
# genotypes = pd.read_csv(genotype_file_path)
# phenotypes = pd.read_table(phenotype_file_path)
genotypes = pd.read_csv("genotype.csv")
phenotypes = pd.read_table("FT10.txt")
print("Loading complete.")

# Change sample IDs to integers and sort columns.
genotypes.columns = np.append(
    pd.Index(["chr", "bp"]), genotypes.columns[2:].astype(int)
)
snp_id = genotypes.chr.astype(str) + '_' + genotypes.bp.astype(str)
snp_id.name = "snp"
genotypes = pd.concat(
    [snp_id, genotypes.iloc[:, 2:].sort_index(axis = 1)],
    axis = 1
)
phenotypes.columns = ["id", "pt"]

# Filter to make sure sample IDs are:
# - Present in both the genotype and phenotype dataframes.
# - Not NA
gt_samples = genotypes.columns[1:].values.astype(int)
pt_samples = phenotypes.id.values
# gt_unique = np.setdiff1d(gt_samples, pt_samples) # nothing here
pt_unique = np.setdiff1d(pt_samples, gt_samples)
gt_drop_cols = phenotypes[
    ~phenotypes.id.isin(pt_unique) & phenotypes.pt.isna()
].id.values
genotypes.drop(columns = gt_drop_cols, inplace = True)
# genotypes.dropna(inplace = True) # nothing here
phenotypes = phenotypes[~(phenotypes.id.isin(pt_unique) | phenotypes.pt.isna())]
print("Filtering complete.")

# Format GWAS outputs.
gwas_output_df = genotypes.loc[:, ["snp"]]
gwas_output_df["p"] = 1.0
gwas_output_df["r2_adj"] = 0.0
print("GWAS output formatting complete.")

# to drop
snps_to_drop = []


def __iter_snp(snp: pd.Series, phenotype: np.ndarray) -> pd.Series:
    """
    For a given SNP, encode the major allele as 0 and the minor allele as 1.
    Perform ordinary least squares linear regression on Phenotype ~ SNP Data.
    Put calculated P and R-Squared_adjusted values into the output dataframe.

    Parameters
    ----------
    snp : pd.Series
        pandas Series of SNP row from processed genotype dataframe.

    phenotype : np.ndarray
        numpy ndarray of phenotype values from processed phenotype dataframe.
    """
    snp_idx = snp.name
    # Encode
    alleles = snp.unique()
    allele_dict = {}
    if len(alleles) == 1:
        snps_to_drop.append(snp_idx)
        return
    else:
        indv = len(snp)
        a0 = (snp == alleles[0]).sum()
        a1 = indv - a0
        if a0 > a1:
            allele_dict = {alleles[0]:0, alleles[1]:1}
        else:
            allele_dict = {alleles[1]:0, alleles[0]:1}
    snp.replace(allele_dict, inplace = True)
    genotypes.loc[snp_idx].replace(allele_dict, inplace = True)
    model = sm.OLS(phenotype, snp.values)
    results = model.fit()
    gwas_output_df.iat[snp_idx, 1] = results.pvalues
    gwas_output_df.iat[snp_idx, 2] = results.rsquared_adj


print("Iterating...")
unused_output = genotypes.iloc[:, 1:].apply(
    func = __iter_snp,
    axis = 1,
    phenotype = phenotypes.pt.values
)
del unused_output #mem management

# Format linear model output.
genotypes = pd.concat([gwas_output_df.p, genotypes], axis = 1)
genotypes.drop(index = snps_to_drop, inplace = True)
genotypes.index = genotypes.snp
genotypes.drop(columns = ["snp"], inplace = True)
genotypes.sort_values(by = 'p', inplace = True)
genotypes.drop(columns = ['p'], inplace = True)
genotypes = genotypes.head(200)
genotypes = genotypes.transpose()
genotypes.index.name = "id"
phenotypes.index = phenotypes.id
phenotypes.drop(columns = ["id"], inplace = True)
lm_output_df = pd.concat([phenotypes, genotypes], axis = 1)
print("Linear model output formatting complete.")

# Sort SNPs ascending by p value.
gwas_output_df.sort_values(by = 'p', inplace = True)
gwas_output_df.head(200).to_csv("gwas_results.csv", index = False)
lm_output_df.to_csv("gt_pt_lm.csv")
print("Done!")
