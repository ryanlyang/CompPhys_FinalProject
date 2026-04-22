# Aspen Shift Sanity Checks (5 Seeds)

- Results root: `/home/ryreu/atlas/CompPhys_FinalProject/restart_studies/results`
- Output dir: `/home/ryreu/atlas/CompPhys_FinalProject/restart_studies/results/prelim_reimpl_cluster_aspen_sanity_5seeds`
- Runs found: `5` / 5

## Pass Counts

- clean_acc_ok: 5/5
- clean_auc_ok: 5/5
- corruption_replay_ok: 5/5
- mapping_strong_r2_ok: 5/5
- mapping_strong_intercept_ok: 5/5
- mapping_strong_loocv_ok: 0/5
- aspen_finite_ok: 5/5
- aspen_norm_ok: 5/5
- aspen_size_ok: 5/5

## Key Aggregate Stats

- clean_acc_diff_stats: mean=0.0, std=0.0, min=0.0, max=0.0
- clean_auc_diff_stats: mean=0.0, std=0.0, min=0.0, max=0.0
- ensemble_pred_delta_stats: mean=0.6324711485246086, std=0.06876334006727595, min=0.5396010759942695, max=0.7040842260596186
- ensemble_pred_acc_stats: mean=0.11345018480872471, std=0.06350510583748677, min=0.04744244060704805, max=0.19727225733906384
- strong_pred_delta_stats: mean=0.48709728857179646, std=0.04111685482597024, min=0.4403437319158096, max=0.5375004995003786
- strong_pred_acc_stats: mean=0.2588240447615368, std=0.0453495858742084, min=0.1993728338329548, max=0.3147962680841904
- top1_max_share_stats: mean=0.7317376, std=0.10792353467293407, min=0.551198, max=0.833619
- num_metrics_clipped_stats: mean=1.4, std=0.8944271909999159, min=0.0, max=2.0

- Extrapolation warnings: warn_total=7, severe_total=18

## Per-Seed Snapshot

- seed 41: status=ok, clean_acc_diff=0.000000, replay_ok=True, severe_extrap_metrics=4, ensemble_pred_acc=0.161378, strong_pred_acc=0.288701
- seed 52: status=ok, clean_acc_diff=0.000000, replay_ok=True, severe_extrap_metrics=3, ensemble_pred_acc=0.068532, strong_pred_acc=0.258233
- seed 63: status=ok, clean_acc_diff=0.000000, replay_ok=True, severe_extrap_metrics=4, ensemble_pred_acc=0.092626, strong_pred_acc=0.314796
- seed 74: status=ok, clean_acc_diff=0.000000, replay_ok=True, severe_extrap_metrics=4, ensemble_pred_acc=0.047442, strong_pred_acc=0.233017
- seed 85: status=ok, clean_acc_diff=0.000000, replay_ok=True, severe_extrap_metrics=3, ensemble_pred_acc=0.197272, strong_pred_acc=0.199373
