python -u 01_train_target_shadow.py
python -u 02_distillation.py
python -u 03_labelonly_normal.py
python -u 04_labelonly_mfa.py --perturb-type target
python -u 04_labelonly_mfa.py --perturb-type distill
python -u 03_shadmodel_normal.py
python -u 04_shadmodel_mfa.py --perturb-type target
python -u 04_shadmodel_mfa.py --perturb-type distill
python -u 03_topconf_normal.py
python -u 04_topconf_mfa.py --perturb-type target
python -u 04_topconf_mfa.py --perturb-type distill
