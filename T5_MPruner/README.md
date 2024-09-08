# Script T5 Pruner 
 
## prepare env
```bash
conda create -n t5_mpruner python=3.9
conda activate t5_mpruner 

cd AAAI_MPruner/T5_MPruner
pip install -r requirements.txt
python prepare.py
```

## first experiment

```bash
CUDA_VISIBLE_DEVICES=1 python T5_MPruner.py --model_path=anonymous78784949/t5-squad-QG \
 --tokenizer_path=tokenizer \
  --useHGModel=true \
  --mpruned_model_path=iter1/mpruned1.pth
  
```

## second experiment

```bash
CUDA_VISIBLE_DEVICES=1 python T5_MPruner.py --model_path=iter1/mpruned1.pth \
 --tokenizer_path=tokenizer \
  --useHGModel=false \
  --mpruned_model_path=iter2/mpruned2.pth
```