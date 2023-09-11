baseline -> without_aux
min_cap -> 1692539407.659011_min_acc_min_cap
min_cap_view -> 1692539431.9839504_min_acc_min_cap_view
view_contrastive -> 1692800777.0677733_min_acc_min_cap_view_contrastive
multiview_cap -> 1692800780.1965835_mean_acc_min_cap_multiview_cap
multiview_cap_view -> 1693415107.6382463_min_acc_min_cap_multiview_cap_view
multiview_view -> 1693820974.163681_mean_acc_min_cap_multiview_view


## 1692539431.9839504_min_acc_min_cap_view
pred:
`[0.7036082474226805, 0.7044673539518901, 0.7036082474226805, 0.7036082474226805, 0.7044673539518901, 0.7044673539518901, 0.7044673539518901, 0.7036082474226805, 0.7036082474226805, 0.7044673539518901]`
random:
`[0.6125429553264605, 0.6262886597938144, 0.6348797250859106, 0.6099656357388317, 0.6237113402061856, 0.6417525773195877, 0.6185567010309279, 0.602233676975945, 0.6185567010309279, 0.647766323024055]`

## 1693415107.6382463_min_acc_min_cap_multiview_cap_view
pred:
`[0.7268041237113402, 0.7268041237113402, 0.7268041237113402, 0.7268041237113402, 0.7268041237113402, 0.7268041237113402, 0.7268041237113402, 0.7268041237113402, 0.7259450171821306, 0.7268041237113402]`
random:
`[0.7130584192439863, 0.7156357388316151, 0.7250859106529209, 0.7079037800687286, 0.7139175257731959, 0.7139175257731959, 0.7259450171821306, 0.729381443298969, 0.7036082474226805, 0.7310996563573883]`

## 1692539431.9839504_min_acc_min_cap_view
{0: 928, 1: 0,  2: 0,   3: 0,   4: 77,  5: 0,   6: 21,  7: 138}
{0: 308, 1: 51, 2: 152, 3: 69,  4: 182, 5: 62,  6: 133, 7: 207}

## 1693415107.6382463_min_acc_min_cap_multiview_cap_view
{0: 556, 1: 0, 2: 0, 3: 0, 4: 89, 5: 0, 6: 0, 7: 519}
{0: 172, 1: 163, 2: 145, 3: 134, 4: 114, 5: 115, 6: 123, 7: 198}


zwar scheint der caption view predictor auf die front views zu fitten, so scheint es jedoch in etwa mit der distribution der min views übereinzustimmen
es könnte auch ein data bias sein
selbst wenn nicht, dann lässt sich mit der view prediction zusammen sinnvolle aussagen treffen..



A white cabinet with 2 doors view: 1 (pred: 1) best view: 7 (pred 7); id: ('ef3b459ecb092dc5738e43095496b061',) 
distractor id: ('b5e8e3356b99a8478542f96306060db4',)
<bos> a train <eos> 

<bos> white cabinet with door <eos> 



medium brown dresser view: 5 (pred: 5) best view: 0 (pred 0); id: ('4ef699bfb20dcb1ac00fd1150223027',) 
distractor id: ('aea6f0ddc6450d490546f5d45015351',)
<bos> a brown cardboard box <eos> 

<bos> brown dresser <eos> 


## BAD
tall grey cabinet with a brown front view: 2 (pred: 2) best view: 0 (pred 0); id: ('3d95c4ebb88f94b3d2f054448bd437d6',) 
distractor id: ('8753d446360545aeb554b42ae3ae3fa0',)
<bos> a black chair <eos> 

<bos> a green chair <eos> 




## spatial cues in annotations:
validation: 35 / 1164
train: 406 / 20085