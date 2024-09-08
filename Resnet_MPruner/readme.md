# Resnet

### Script Resnet Pruner

---

> Prepare env
> 

---

```bash
conda activate -n resnet_mpruner python=3.11.8
conda activate resnet_mpruner

cd AAAI_MPruner/Resnet_MPruner
pip install -r requirement.txt
```

> Experiment of resnet152
> 

---

```bash
CUDA_VISIBLE_DEVICES=0 python Resnet_MPruner.py 
--useHGModel True \
--load anonymous78784949/resnet152-imagenet1k \
--batch_size 64 \
--hook ResNetBottleNeckLayer \
--threshold 98 \
--acc_threshold 4 \
--epoch 3 \
--iterate 1 \
--freeze False \
--save ./result
```

> Experiment of resnet50
> 

---

```bash
CUDA_VISIBLE_DEVICES=0 python Resnet_MPruner.py 
--useHGModel True \
--load anonymous78784949/resnet50-imagenet1k \
--batch_size 64 \
--hook ResNetBottleNeckLayer \
--threshold 98 \
--acc_threshold 4 \
--epoch 3 \
--iterate 1 \
--freeze False \
--save ./result
```