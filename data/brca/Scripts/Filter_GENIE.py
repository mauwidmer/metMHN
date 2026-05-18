import pandas as pd
from tqdm import tqdm
tqdm.pandas()

# ==============================
# USER SETTINGS
# ==============================
INPUT_FILE = "annotated_mutations.txt"
OUTPUT_FILE = "filtered_mutations.txt"

ONCOTREE_CODES = ["BRCA", "ACBC", "BRSRCC", "BRCANOS", "BRCNOS", "CSNOS", "IDC", "ILC", "IMMC", "MDLC", "SPC"]   
# change to ["LUAD"] or ["COAD", "LUAD"]
# for breast ["BRCA"]

# Optional: provide MSK-IMPACT gene list
MSK_IMPACT_GENES_FILE = None  # e.g. "msk_impact_genes.txt"

# ==============================
# LOAD DATA
# ==============================
print("loading data")

# Load mutation data
df = pd.read_csv("../Data/GENIE19/"+INPUT_FILE, sep="\t", low_memory=False)

# Load clinical sample data
clinical = pd.read_csv(
    "../Data/GENIE19/data_clinical_sample.txt",
    sep="\t",
    comment="#"
)

# Merge
df = df.merge(
    clinical[["SAMPLE_ID", "ONCOTREE_CODE", "AGE_AT_SEQ_REPORT"]],
    left_on="Tumor_Sample_Barcode",
    right_on="SAMPLE_ID",
    how="left"
)

df["AGE_AT_SEQ_REPORT"] = pd.to_numeric(
    df["AGE_AT_SEQ_REPORT"], errors="coerce"
)

print("After merge:", df.shape)
print("Missing Oncotree:", df["ONCOTREE_CODE"].isna().sum())
print("loading data complete")
# ==============================
# FILTER: Oncotree
# ==============================
df = df[df["ONCOTREE_CODE"].isin(ONCOTREE_CODES)]

# ==============================
# OPTIONAL: select one sample per patient (lowest age at sequencing)
# ==============================
if "PATIENT_ID" in df.columns and "Age_at_Sequencing" in df.columns:
    df = df.sort_values("Age_at_Sequencing")
    df = df.groupby("PATIENT_ID").first().reset_index()

# ==============================
# FILTER STEP 1: OncoKB
# ==============================
def filter_oncokb(row):
    val = str(row.get("ONCOGENIC", "")).strip()
    
    if val in ["Oncogenic", "Likely Oncogenic", "Resistance"]:
        return True
    if val in ["Neutral", "Likely Neutral"]:
        return False
    return None  # go to step 2
print("Filter 1 defined")

# ==============================
# FILTER STEP 2: PolyPhen + SIFT
# ==============================
def filter_polyphen_sift(row):
    poly = str(row.get("Polyphen_Prediction", "")).lower()
    sift = str(row.get("SIFT_Prediction", "")).lower()

    if "probably_damaging" in poly:
        return True
    if "possibly_damaging" in poly and (
        "deleterious" in sift or sift == "" or "low_confidence" in sift
    ):
        return True
    if poly == "" and "deleterious" in sift:
        return True

    # if either annotation exists but doesn't pass → reject
    if poly != "" or sift != "":
        return False

    return None  # go to step 3
print("Filter 2 defined")

# ==============================
# FILTER STEP 3: Variant Classification
# ==============================
ACCEPTED_VARIANTS = {
    "Frame_Shift_Ins",
    "Frame_Shift_Del",
    "Nonsense_Mutation",
    "Nonstop_Mutation",
    "Splice_Site",
    "Translation_Start_Site",
}

def filter_variant_class(row):
    vc = str(row.get("Variant_Classification", ""))
    return vc in ACCEPTED_VARIANTS
print("Filter 3 defined")

# ==============================
# APPLY FILTER CASCADE
# ==============================
def apply_filter(row):
    step1 = filter_oncokb(row)
    if step1 is not None:
        return step1

    step2 = filter_polyphen_sift(row)
    if step2 is not None:
        return step2

    return filter_variant_class(row)

# df["KEEP"] = df.apply(apply_filter, axis=1)
df["KEEP"] = df.progress_apply(apply_filter, axis=1)
df_filtered = df[df["KEEP"]]
df["KEEP"] = df["KEEP"].map({True: "TRUE", False: "FALSE"}) # makes it readable for R

# ==============================
# OPTIONAL: MSK-IMPACT genes
# ==============================
if MSK_IMPACT_GENES_FILE:
    genes = set(pd.read_csv(MSK_IMPACT_GENES_FILE, header=None)[0])
    df_filtered = df_filtered[df_filtered["Hugo_Symbol"].isin(genes)]

# ==============================
# SAVE
# ==============================
df_filtered.to_csv("../Data/GENIE19/"+OUTPUT_FILE, sep="\t", index=False)

print("Done!")
print(f"Input variants: {len(df)}")
print(f"Filtered variants: {len(df_filtered)}")