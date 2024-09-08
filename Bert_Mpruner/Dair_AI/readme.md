# Dair AI Emotion

### Script Dair AI Emotion Pruner

---

> Prepare env
> 

---

```bash
conda activate -n emotion_mpruner python=3.11.8
conda activate emotion_mpruner

cd AAAI_MPruner/Emotion_MPruner
cd Dair_AIt
pip install -r requirement.txt
```

> Experiment of Dair AI Emotion
> 

---

```bash
CUDA_VISIBLE_DEVICES=0 python Emotion_MPruner.py 
--useHGModel True \
--load bhadresh-savani/bert-base-uncased-emotion \
--data_name emotion \
--hook BeitLayer \
--threshold 98 \
--acc_threshold 4 \
--epoch 3 \
--iterate 1 \
--freeze False \
--save ./result
```
