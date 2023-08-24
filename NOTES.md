# baseline snare test
min:    0.70
mean:   0.69
min per view mean:      0.6941580756013745
view pred:      0.12177835404872894
cap view pred:  0.1168384924530983

train the first two again, but on all training data!
## large_train_no_freeze_min_cap_view_view_contrastive
pred:   0.70
random: 0.61

## large_train_no_freeze_min_cap_view
pred:   0.70
random: 0.67

## baseline
random: ~0.66




## larger_train_no_freeze_multiview_cap_min_cap_view_view_contrastive, 10 runs
predicted:

`[0.7551546391752577, 0.7551546391752577, 0.7542955326460481, 0.7542955326460481, 0.7551546391752577, 0.7551546391752577, 0.7551546391752577, 0.7551546391752577, 0.7551546391752577, 0.7551546391752577]`

random:

`[0.7250859106529209, 0.7362542955326461, 0.7233676975945017, 0.7362542955326461, 0.7474226804123711, 0.7353951890034365, 0.7345360824742269, 0.7310996563573883, 0.7207903780068728, 0.7362542955326461]`


baseline -> without_aux
min_cap -> 1692539407.659011_min_acc_min_cap
min_cap_view -> 1692539431.9839504_min_acc_min_cap_view
view_contrastive -> 1692800777.0677733_min_acc_min_cap_view_contrastive
multiview_cap -> 1692800780.1965835_mean_acc_min_cap_multiview_cap