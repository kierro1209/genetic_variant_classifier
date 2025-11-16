This project details a genetic pathogenicity classifier using XGBoost models to predict based on chromosomal location, variant length,
clinical reviews, frequency of the variation, and type of variation.

## Overview:

Data Sources: 
1. ClinVar - https://www.ncbi.nlm.nih.gov/clinvar/

Models
1. LASSO
2. PCA
3. XGBoost

Methods:
During feature engineering to acquire more features that are biologically related to pathogenicity, such as variation length, origin, variation type etc., I downsampled from the 3.8M variants in the GCRh38 assembly, to transform 600,000. Then from there, I downsampled randomly again to address the class imbalance within the dataset of variants labelled as non-pathogenic, so I had a more even 60-40 split of non-pathogenic to pathogenic variants used. Then, I used LASSO and PCA to understand if dimensionality reduction was a possibility before modelling, but PCA revealed that the explained variance was too low amongst the differenct principal components to warrants removing any features. Then I trained an XGBoost model using an 80-20 train-test split getting a [accuracy rate] and validated the model's accuracy using the [method] and getting a [metric] of [rate].

Kiersten Roth
UCLA Statistics & Data Science
Connect: https://www.linkedin.com/in/kiersten-roth/ 









