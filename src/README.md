[TOC]

# QQ Similarity

## Result

### 20190715

| 模型     | 指标            | AUC        | ACC        | PRE        | REC        | F1         | 阈值 | 备注             |
| -------- | --------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---- | ---------------- |
| FastText | QQ Pair级别指标 | 0.8007     | 0.7611     | 0.5982     | 0.6780     | 0.6356     | 0.33 | 最高F1，30 epoch |
| FastText | QQ Pair级别指标 | 0.8007     | 0.7669     | 0.6174     | 0.6349     | 0.6261     | 0.8  | 高阈值，30 epoch |
| FastText | QQ Pair级别指标 | 0.8007     | **0.7694** | **0.6272** | 0.6157     | 0.6214     | 0.9  | 高阈值，30 epoch |
| TextCNN  | QQ Pair级别指标 | 0.7226     | 0.6203     | 0.4360     | 0.8025     | 0.5650     | 0.33 | 最高F1，28 epoch |
| TextCNN  | QQ Pair级别指标 | 0.7226     | 0.6501     | 0.4562     | 0.7216     | 0.5590     | 0.8  | 高阈值，28 epoch |
| TextCNN  | QQ Pair级别指标 | 0.7226     | 0.6617     | 0.4654     | 0.6799     | 0.5526     | 0.9  | 高阈值，28 epoch |
| SCWLSTM  | QQ Pair级别指标 | **0.8202** | 0.7328     | 0.5446     | 0.7953     | **0.6465** | 0.91 | 最高F1，11 epoch |
| SCWLSTM  | QQ Pair级别指标 | **0.8202** | 0.7178     | 0.5261     | **0.8253** | 0.6426     | 0.8  | 高阈值，11 epoch |
| SCWLSTM  | QQ Pair级别指标 | **0.8202** | 0.7311     | 0.5423     | 0.7995     | 0.6463     | 0.9  | 高阈值，11 epoch |
| APLSTM   | QQ Pair级别指标 | 0.7076     | 0.6143     | 0.4283     | 0.7621     | 0.5484     | 0.17 | 最高F1，9 epoch  |
| APLSTM   | QQ Pair级别指标 | 0.7076     | 0.7077     | 0.5342     | 0.3817     | 0.4452     | 0.8  | 高阈值，9 epoch  |
| APLSTM   | QQ Pair级别指标 | 0.7076     | 0.7107     | 0.5570     | 0.2865     | 0.3784     | 0.9  | 高阈值，9 epoch  |

### 20190719





## Model

### Siamese LSTM

[Siamese Recurrent Architectures for Learning Sentence Similarity](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf)

[How to predict Quora Question Pairs using Siamese Manhattan LSTM](https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07)

### APLSTM/APCNN

### ARCII

### BiMPM