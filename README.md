# Deeparc 제출

---

# Resnet

- hardware :
    - CPU : In-tel(R) Xeon(R) W5-2455X processor
    - GPU :  NVIDIA RTX A4000 16GB GPU
    - MEMORY : 128 GB
- Dataset :  ILSVRC-2012
- Model :
    - microsoft/resnet152
    - microsorf/restnet50

## Resnet152

해당 실험은 microsoft/resnet152를 3 epoch으로 fine-tuning한 모델과 해당 모델을 MPruning을 한 모델을 비교한 실험이다.

본 실험은 threshold 98, 99를 주었고, 각각 3 epoch retraining과 freeze를 해보았다. batch size는 64를 주었으며, optimizer는 adam, learning rate는 0.0002이다. 해당 모델 layer pruning 과정 중 ResNetBottleNeck Layer를 기준으로 Layer 유사성을 바탕으로 진행하였다. 실험의 결과는 아래와 같다.

### Original

|  | 1epoch | 2epoch | 3epoch | Num. of layers | Parameter | Evaluation time(s) |
| --- | --- | --- | --- | --- | --- | --- |
| Resnet152 | N/A | N/A | 76.824 | 50 | 60.192808 M | 259.9679s |

### All [1st iteration] K = 1

|  | 1epoch | 2epoch | 3epoch | Num. of layers | Parameter | Evaluation time(s) | freeze |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Threshold99 | 72.094 | 72.926 | 73.374 (-3.45%) | 26 | 31.709224 M | 180.18139s | 0 |
| Threshold98 | 70.318 | 72.0 | 72.164 (-4.66%) | 22 | 27.240488 M | 169.60528s | 0 |
| Threshold99(Part) | 72.604 | 73.494 (-3.33%) | 73.176 (-4.33%) | 26 | 31.709224 M | 180.17326s | 16 |
| Threshold98(Part) | 70.826 | 71.894 | 72.294 (-4.53%) | 22 | 27.240488 M | 169.92417s | 15 |

### Half (Odd) [1st iteration] K = 2

|  | 1epoch | 2epoch | 3epoch | Num. of layers | Parameter | Evaluation time(s) | freeze |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Threshold99 | 74.758 | 74.960 (-1.864%) | 74.622 | 33 | 39.529512 M | 202.43821s | 0 |
| Threshold98 | 74.768 | 74.926 (-1.898%) | 74.652 | 32 | 38.412328 M | 201.78150s | 0 |
| Threshold99(Part) | 74.572 | 74.956 (-1.868%) | 74.818 | 33 | 39.529512 M | 202.33958s | 9 |
| Threshold98(Part) | 74.51 | 74.688 (-2.136%) | 74.6859 | 32 | 38.412328 M | 201.15831s | 8 |

### All [2nd iteration] K = 1

|  | 1epoch | 2epoch | 3epoch | Num. of layers | Parameter | Evaluation time(s) | freeze |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Threshold99 | 70.226 | 70.39 | 70.53 | 20 | 25.84324 M | 162.24440s | 0 |
| Threshold98 | 67.168 | 68.49 | 68.822 | 14 | 19.977256 M | 139.80341s | 0 |
| Threshold99(Part) | 68.984 | 69.512 | 70.054 | 17 | 22.491688 M | 151.83072s | 9 |
| Threshold98(Part) | 70.35 | 70.61 | 70.254 | 17 | 23.328808 M | 149.61890s | 9 |

### Half (Odd) [2nd iteration] K= 2

|  | 1epoch | 2epoch | 3epoch | Num. of layers | Parameter | Evaluation time(s) | freeze |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Threshold99 | 73.02 | 73.38 | 73.106 | 26 | 33.663528 M | 181.37359s | 0 |
| Threshold98 | 72.44 | 72.789 | 72.807 | 23 | 30.031912 M | 166.74806s | 0 |
| Threshold99(Part) | 73.672 | 73.842 | 73.64 | 27 | 33.663528 M | 181.26035s | 15 |
| Threshold98(Part) | 72.326 | 72.460 | 72.64 | 22 | 28.077608 M | 168.01924s | 8 |

## Resnet 50

해당 실험은 microsoft/resnet50을 3epoch으로 fine-tuning한 모델과 해당 모델을 MPruning을 한 모델을 비교한 실험이다.

본 실험은 threshold 95, 98, 99를 주었고, 그 중 99는 Pruning 되는 것이 없어 제외하였다. 마찬가지로 각각 3 epoch retraining과 freeze를 해보았다. batch size는 64를 주었으며, optimizer는 adam, learning rate는 0.0002이다. 해당 모델 layer pruning 과정 중 ResNetBottleNeck Layer를 기준으로 Layer 유사성을 바탕으로 진행하였다. 실험의 결과는 아래와 같다.

### Original

|  | 1epoch | 2epoch | 3epoch | Num. of layers | Parameter | Evaluation time(s) |
| --- | --- | --- | --- | --- | --- | --- |
| Resnet50 | 77.2 | 77.082 | 76.754 | 16 | 25.557032 M | 147.0041 |

### All [1st iteration] K = 1

|  | 1epoch | 2epoch | 3epoch | Num. of layers | Parameter | Evaluation time(s) | freeze |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Threshold99 | (=original) |  |  |  |  |  |  |
| Threshold98 | 75.698(-1.056%) | 75.44 | 75.2719 | 14 | 24.159784 M | 136.8380 | 0 |
| Threshold95 | 72.202 | 72.452(-4.302%) | 72.432 | 10 | 21.574952 M | 100.2376 | 0 |
| Threshold98(Part) | 75.71(-1.044%) | 75.504 | 75.2 | 14 | 24.159784 M | 136.717 | 10 |
| Threshold95(Part) | 72.289 | 72.5219(-4.2321%) | 72.344 | 10 | 21.574952 M | 118.4091 | 4 |

### All [2nd iteration] K = 1

|  | 1epoch | 2epoch | 3epoch | Num. of layers | Parameter | Evaluation time(s) | freeze |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Threshold98 | 74.75 | 74.652 | 74.468 | 13 | 23.87972 M | 130.574 | 0 |
| Threshold95 | 70.414 | 70.69 | 70.524 | 8 | 21.224488 M | 110.483 | 0 |
| Threshold98(Part) | 72.48 | 72.568 | 72.658 | 11 | 21.294888 M | 127.497 | 9 |
| Threshold95(Part) | 71.7539 | 71.364 | 71.566 | 9 | 21.645352 M | 116.2409 | 5 |

## Dair AI Emotion

- hardware
    - CPU : 13th Gen Intel(R) Core(TM) i7-13700
    - GPU : NVIDIA GeForce RTX 3070 TI 8GB
    - MEMORY : 16GB
- Dataset
    - Dair AI Emotion
- Model
    - Bert-base / text classification

해당 실험은 Bert-base / text classification을 1 epoch으로 fine-tuning한 모델과 해당 모델을 MPruing을 한 모델을 비교한 실험이다. 또한 Wanda를 MPruning한 모델에 대해서 Sparsity를 각 10, 20, 30, 40, 50% 주어 비교하였다.

본 실험은 threshold 98, 99를 주었고, 각각 3 epoch retraining과 freeze를 해보았다. batch size는 32를 주었으며, optizmizer는 AdamW, learning rate는 5e-05이다. 해당 모델 layer pruning 과정 중 Encoder를 기준으로 BertLayer 유사성을 바탕으로 진행하였다. 실험의 결과는 아래와 같다. 

### Original

|  | Num. of encoders | Evaluation loss | Accuracy | Evaluation time(s) | Training time(s) | Parameter |
| --- | --- | --- | --- | --- | --- | --- |
| Bert | 12 | 0.3312 | 0.92 | 14.8684 | 1579.8736 | 109.486854 M |

### All [1st iteration]

|  | Num. of encoders | Evaluation loss | Accuracy | Evaluation time(s) | Training time(s) | Parameter | freeze |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Threshold99 | 10 | 0.41597 | 0.921(+0.001%) | 8.6455 | 1306.872 | 95.31111 M | 0 |
| Threshold98 | 8 | 0.3665 | 0.934(+0.014%) | 7.0948 | 1062.2365 | 81.135366 M | 0 |
| Threshold99(Part) | 10 | 0.31247 | 0.926(+0.006%) | 8.6366 | 1126.2001 | 95.31111 M | 6 |
| Threshold98(Part) | 8 | 0.3341 | 0.925(+0.005%) | 6.9947 | 929.7921 | 81.135366 M | 4 |

### All [2nd iteration]

|  | Num. of encoders | Evaluation loss | Accuracy | Evaluation time(s) | Training time(s) | Parameter | freeze |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Threshold98 | 6 | 0.4054 | 0.928(+0.008%) | 5.5214 | 796.2302 | 66.959622 M | 0 |
| Threshold98(Part) | 6 | 0.41388794 | 0.915(-0.005%) | 5.3583 | 706.6606 | 66.959622 M | 3 |

### Wanda Sparsity Original

|  | Evaluation loss before training | Accuracy before training | Evaluation time(s) | Parameter |
| --- | --- | --- | --- | --- |
| 0.1 | 0.23041 | 0.949(+0.029%) | 10.272 | 109.486854M |
| 0.2 | 0.28538 | 0.921 | 10.382 | 109.486854M |
| 0.3 | 0.308224 | 0.921 | 10.116 | 109.486854M |
| 0.4 | 0.26523 | 0.928 | 10.0158 | 109.486854M |
| 0.5 | 0.2619 | 0.933 | 9.9735 | 109.486854M |

### Deeparc with Wanda Sparsity 10%

|  | Evaluation loss before training | Accuracy before training | Evaluation time(s) | Parameter |
| --- | --- | --- | --- | --- |
| Threshold98[1차] | 0.38772 | 0.931 | 6.9494 | 81.135366M |
| Threshold98(Part)[1차] | 0.3032 | 0.928 | 6.6892 | 81.135366M |
| Threshold98[2차] | 0.4431 | 0.921 | 5.3029 | 66.959622M |
| Threshold98(Part)[2차] | 0.3947 | 0.925 | 5.129 | 66.959622M |

### Deeparc with Wanda Sparsity 20%

|  | Evaluation loss before training | Accuracy before training | Evaluation time(s) | Parameter |
| --- | --- | --- | --- | --- |
| Threshold98[1차] | 0.3383 | 0.933 | 6.7985 | 81.135366M |
| Threshold98(Part)[1차] | 0.3045 | 0.936 | 6.7756 | 81.135366M |
| Threshold98[2차] | 0.5117 | 0.911 | 5.2262 | 66.959622M |
| Threshold98(Part)[2차] | 0.4068 | 0.918 | 5.2201 | 66.959622M |

### Deeparc with Wanda Sparsity 30%

|  | Evaluation loss before training | Accuracy before training | Evaluation time(s) | Parameter |
| --- | --- | --- | --- | --- |
| Threshold98[1차] | 0.3808 | 0.928 | 6.8346 | 81.135366M |
| Threshold98(Part)[1차] | 0.2683 | 0.942 | 6.824 | 81.135366M |
| Threshold98[2차] | 0.3969 | 0.923 | 5.1384 | 66.959622M |
| Threshold98(Part)[2차] | 0.3494 | 0.927 | 5.1097 | 66.959622M |

### Deeparc with Wanda Sparsity 40%

|  | Evaluation loss before training | Accuracy before training | Evaluation time(s) | Parameter |
| --- | --- | --- | --- | --- |
| Threshold98[1차] | 0.3836 | 0.921 | 6.7959 | 81.135366M |
| Threshold98(Part)[1차] | 0.3071 | 0.931 | 6.737 | 81.135366M |
| Threshold98[2차] | 0.4336 | 0.918 | 5.1144 | 66.959622M |
| Threshold98(Part)[2차] | 0.3540 | 0.925 | 5.1172 | 66.959622M |

### Deeparc with Wanda Sparsity 50%

|  | Evaluation loss before training | Evaluation loss after training | Accuracy before training | Accuracy after training | Evaluation time(s) | Training time(s) | Parameter |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Threshold98[1차] | 0.2765 | 0.3679 | 0.937 | 0.942 | 7.0152 | 1052.6543 | 81.135366M |
| Threshold98(Part)[1차] | 0.2304 | 0.3802 | 0.94 | 0.927 | 6.9785 | 936.1501 | 81.135366M |
| Threshold98[2차] | 0.3805 | 0.4955 | 0.926 | 0.921 | 5.3181 | 803.2597 | 66.959622M |
| Threshold98(Part)[2차] | 0.3390 | 0.4493 | 0.929 | 0.925 | 5.3052 | 711.3443 | 66.959622M |

## Yahoo Answers Topics

- hardware
    - CPU : 14th Gen Intel(R) Core(TM) i9-14900
    - GPU : NVIDIA GeForce RTX 4090 24GB
    - MEMORY : 64GB
- Dataset
    - Yahoo answer topics
- Model
    - Bert-base / text classification

해당 실험은 Bert-base / text classification을 1 epoch으로 fine-tuning한 모델과 해당 모델을 MPruing을 한 모델을 비교한 실험이다. 또한 Wanda를 MPruning한 모델에 대해서 Sparsity를 각 10, 20, 30, 40, 50% 주어 비교하였다.

본 실험은 threshold 98, 99를 주었고, 각각 3 epoch retraining과 freeze를 해보았다. batch size는 32를 주었으며, optizmizer는 AdamW, learning rate는 5e-05이다. 해당 모델 layer pruning 과정 중 Encoder를 기준으로 BertLayer 유사성을 바탕으로 진행하였다. 실험의 결과는 아래와 같다.

### Original

|  | Num. of encoders | Evaluation loss | Accuracy | Evaluation time(s) | Training time(s) | Parameter |
| --- | --- | --- | --- | --- | --- | --- |
| Bert | 12 | 1.2287 | 0.7047 | 101.9891 | 41325.0308 | 109.48993 M |

### All [1st iteration]

|  | Num. of encoders | Evaluation loss | Accuracy | Evaluation time(s) | Training time(s) | Parameter | freeze |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Threshold99 | 12 (=original) | - | - | - | - | - | - |
| Threshold98 | 9 | 1.094 | 0.710 | 79.4448 | 31593.5162 | 88.226314 M | 0 |
| Threshold99(Part) | 12 (=original) | - | - | - | - | - | - |
| Threshold98(Part) | 9 | 1.1068 | 0.7099 | 80.0802 | 31687.6199 | 88.226314 M | 4 |

### Wanda Sparsity Original

|  | Evaluation loss before training | Accuracy before training | Evaluation time(s) | Parameter |
| --- | --- | --- | --- | --- |
| 0.1 | 1.0265 | 0.71716 | 101.9313 | 109.48993M |
| 0.2 | 1.0385 | 0.71616 | 102.1489 | 109.48993M |
| 0.3 | 1.03147 | 0.7213 | 102.3629 | 109.48993M |
| 0.4 | 1.01104 | 0.7213 | 101.6302 | 109.48993M |
| 0.5 | 0.9964 | 0.7214 | 101.568 | 109.48993M |

### Deeparc with Wanda Sparsity 10%

|  | Evaluation loss before training | Accuracy before training | Evaluation time(s) | Parameter |
| --- | --- | --- | --- | --- |
| Threshold98 | 1.0982 | 0.7132 | 79.8796 | 88.226314M |
| Threshold98(Part) | 1.0854 | 0.71286 | 79.6928 | 88.226314M |

### Deeparc with Wanda Sparsity 20%

|  | Evaluation loss before training | Accuracy before training | Evaluation time(s) | Parameter |
| --- | --- | --- | --- | --- |
| Threshold98 | 1.0906 | 0.71483 | 79.9597 | 88.226314M |
| Threshold98(Part) | 1.1071 | 0.7119 | 79.7147 | 88.226314M |

### Deeparc with Wanda Sparsity 30%

|  | Evaluation loss before training | Accuracy before training | Evaluation time(s) | Parameter |
| --- | --- | --- | --- | --- |
| Threshold98 | 1.0875 | 0.7126 | 79.6492 | 88.226314M |
| Threshold98(Part) | 1.0877 | 0.71203 | 79.8392 | 88.226314M |

### Deeparc with Wanda Sparsity 40%

|  | Evaluation loss before training | Accuracy before training | Evaluation time(s) | Parameter |
| --- | --- | --- | --- | --- |
| Threshold98 | 1.0789 | 0.71613 | 79.0803 | 88.226314M |
| Threshold98(Part) | 1.0718 | 0.7160 | 79.3074 | 88.226314M |

### Deeparc with Wanda Sparsity 50%

|  | Evaluation loss before training | Accuracy before training | Evaluation time(s) | Parameter |
| --- | --- | --- | --- | --- |
| Threshold98 | 1.0564 | 0.71763 | 79.4558 | 88.226314M |
| Threshold98(Part) | 1.0523 | 0.7183 | 78.9581 | 88.226314M |

## T5 model - SQuAD (QG)

- 실험 환경
    - 하드웨어
        - CPU : intel xeon w5-2455x ( 128 processor )
        - GPU : RTX A6000 48GB
        - MEMORY : 256GB
    - Dataset
        - SQuAD v1.1
    - Model
        - T5-base Question Generation (QG)

해당 실험은 SQuAD v1.1 data-set을 t5-base모델로 Question_Generation 역할을 수행하였다. 

 m-pruning 과정 중 유사도 임계값은 99%로 설정하였다.

 layer pruning 이후 3 epoch의 재학습을 진행해 모델을 미세조정했다. 

실험결과는 아래와 같다.

|  | base-model (T5-base) | 1st_iter (retrain 3epoch) | final_iter |
| --- | --- | --- | --- |
| Num of encoder | 12 | 9 | 7 |
| Num of decoder | 12 | 5 | 3 |
| Num of parameter | 222M (222,903,552) | 135M (135,588,864) | 109M (109,630,464) |
| BLEU | 0.2268 | 0.2120 | 0.2009 |
| Precision 1-gram | 0.5319 | 0.5339 | 0.5296 |
| Precision 2-gram | 0.2809 | 0.2741 | 0.2659 |
| Precision 3-gram | 0.1825 | 0.1760 | 0.1678 |
| Precision 4-gram | 0.1237 | 0.1176 | 0.1103 |
| Brevity Penalty | 0.9411 | 0.9038 | 0.8890 |
| Length Ratio | 0.9411 | 0.9082 | 0.8890 |
| ROUGE-1 F1 | 0.5177 | 0.5027 | 0.4929 |
| ROUGE-2 F1 | 0.3033 | 0.2880 | 0.2765 |
| ROUGE-L F1 | 0.4812 | 0.4697 | 0.4601 |
| ROUGE-Lsum F1 | 0.4812 | 0.4702 | 0.4601 |
| METEOR | 0.4894 | 0.4702 | 0.4590 |
