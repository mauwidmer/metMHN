import metmhn.regularized_optimization as reg_opt
import metmhn.Utilityfunctions as utils
import pandas as pd
import jax.numpy as jnp
import jax.random as jrp
import numpy as np
import jax as jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import logging
# Adapt path to where logs should be kept
logging.basicConfig(filename='../../logs/analysis_example.log',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    filemode='w', 
                    level=logging.INFO, 
                    force=True,
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )

mut_handle = "../data/luad/G14_LUAD_Events.csv"
annot_handle = "../data/luad/G14_LUAD_sampleSelection.csv"
annot_data = pd.read_csv(annot_handle)
mut_data = pd.read_csv(mut_handle)
mut_data.rename(columns={"Unnamed: 0":"patientID"}, inplace = True)
dat = pd.merge(mut_data, annot_data.loc[:, ['patientID', 'metaStatus']],
               on=["patientID", "patientID"])
muts = ['P.TP53 (M)', 'M.TP53 (M)', 'P.KRAS (M)', 'M.KRAS (M)', 'P.EGFR (M)', 'M.EGFR (M)', 'P.STK11 (M)', 'M.STK11 (M)', 'P.KEAP1 (M)', 'M.KEAP1 (M)',
        'P.RBM10 (M)', 'M.RBM10 (M)', 'P.SMARCA4 (M)', 'M.SMARCA4 (M)', 'P.ATM (M)', 'M.ATM (M)', 'P.NF1 (M)', 'M.NF1 (M)', 'P.PTPRD (M)', 'M.PTPRD (M)']
        #For a faster run, we can use only the 10 most frequent mutations. The full set of mutations would include:
        #'P.PTPRT (M)', 'M.PTPRT (M)', 'P.ARID1A (M)', 'M.ARID1A (M)', 'P.BRAF (M)', 'M.BRAF (M)', 'P.PIK3CA (M)', 'M.PIK3CA (M)', 'P.EPHA3 (M)', 'M.EPHA3 (M)',
        #'P.FAT1 (M)', 'M.FAT1 (M)', 'P.SETD2 (M)', 'M.SETD2 (M)', 'P.RB1 (M)', 'M.RB1 (M)', 'P.MET (M)', 'M.MET (M)', 'P.KMT2C (M)', 'M.KMT2C (M)'

# Convert string labels to integer labels
# "Paired" == 0 and  "metastatus" == "absent" -> 0
# "Paired" == 0 and  "metastatus" == "present" -> 1
# "Paired" == 0 and  "metastatus" == "isMetastasis" -> 2
# "Paired" == 1 -> 3
# Else -> pd.NA
dat["type"] = dat.apply(utils.categorize, axis=1)
# Add the seeding event
dat["Seeding"] = dat["type"].apply(lambda x: pd.NA if pd.isna(x) else 0 if x == 0 else 1)
dat["M.AgeAtSeqRep"] = pd.to_numeric(dat["M.AgeAtSeqRep"], errors='coerce')
dat["P.AgeAtSeqRep"] = pd.to_numeric(dat["P.AgeAtSeqRep"], errors='coerce')
# Define the order of diagnosis for paired datapoints
dat["diag_order"] = dat["M.AgeAtSeqRep"] - dat["P.AgeAtSeqRep"]
dat["diag_order"] = dat["diag_order"].apply(lambda x: pd.NA if pd.isna(x) else 2 if x < 0 else 1 if x > 0 else 0) 
dat["diag_order"] = dat["diag_order"].astype(pd.Int64Dtype())

events_data = muts+["Seeding"]

# Only use datapoints where the state of the seeding is known
cleaned = dat.loc[~pd.isna(dat["type"]), muts+["Seeding", "diag_order", "type"]]
dat = jnp.array(cleaned.to_numpy(dtype=np.int8, na_value=-99))

events_plot = []
for elem in cleaned.columns[:-3].to_list()[::2]:
    full_mut_id = elem.split(".")
    events_plot.append(full_mut_id[1])
events_plot.append("Seeding")

n_tot = (cleaned.shape[1]-1)//2 + 1
n_mut = n_tot-1
utils.marg_frequs(dat, events_plot)

w_corr = 0.65

log_lams = np.linspace(-3.5, -2.5, 5)
lams = 10**log_lams
key = jrp.key(42)
penal_weights = utils.cross_val(dat=dat,
                               penal_fun=reg_opt.symmetric_penal,
                               splits=lams,
                               n_folds=5,
                               m_p_corr=w_corr,
                               key = key)

# The cross_val function returns a n_folds x log_lams.size shaped dataframe
best_lam = lams[np.argmax(np.mean(penal_weights, axis=0))]

df_cv = pd.DataFrame(penal_weights, columns=lams)
df_cv.to_csv("../results/luad/cross_validation_results_1.csv", index=False)

th_init, dp_init, dm_init = utils.indep(dat)
theta, d_p, d_m= reg_opt.learn_mhn(th_init=th_init,
                                   dp_init=dp_init,
                                   dm_init=dm_init,
                                   dat=dat,
                                   perc_met=0.2,
                                   penal=reg_opt.symmetric_penal,
                                   w_penal=0.001,
                                   fixed_grad = False,
                                   opt_ftol=1e-05
                                   )

# Log final likelihood
final_loglik = reg_opt.score(theta, d_p, d_m, dat, perc_met=0.2)
df_loglik = pd.DataFrame({"final_loglik": [float(final_loglik)]})
df_loglik.to_csv("../results/luad/final_loglik_1.csv", index=False)

th_plot = np.row_stack((d_p.reshape((1,-1)),
                        d_m.reshape((1,-1)),
                        theta))
fig, (ax1, ax2) = plt.subplots(1,2, sharey="col", figsize=(14,9),
                                gridspec_kw={'width_ratios': [n_tot, 1], "wspace": -0.6})
utils.plot_theta(ax1, ax2, th_plot, events_plot, alpha=0.2, font_size=5)
plt.show()

df2 = pd.DataFrame(th_plot, columns=events_plot)
df2.to_csv("../results/luad/luad_g14_10muts_1.csv")
