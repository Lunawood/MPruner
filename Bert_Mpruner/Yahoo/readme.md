# Yahoo

### Script Yahoo Pruner

---

> Prepare env
> 

---

```bash
conda activate -n yahoo_mpruner python=3.11.8
conda activate yahoo_mpruner

cd AAAI_MPruner/Yahoo_MPruner
cd Yahoo
pip install -r requirement.txt
```

> Experiment of Yahoo
> 

---

```bash
CUDA_VISIBLE_DEVICES=0 python Yahoo_MPruner.py 
--useHGModel True \
--load fabriceyhc/bert-base-uncased-yahoo_answers_topics \
--data_name yahoo_answers_topics \
--hook BeitLayer \
--threshold 98 \
--acc_threshold 4 \
--epoch 3 \
--iterate 1 \
--freeze False \
--save ./result
```
