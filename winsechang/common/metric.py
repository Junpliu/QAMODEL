import sys
import pandas as pd
from sklearn import metrics

model_name = sys.argv[1]
thre = float(sys.argv[2])
f = pd.read_csv(model_name, sep = ',')

f.loc[f.score > thre, 'pred'] = 1
f.loc[f.score <= thre, 'pred'] = 0
print(f.describe())
print(metrics.accuracy_score(f.label, f.pred))
print(metrics.precision_score(f.label, f.pred))
print(metrics.recall_score(f.label, f.pred))
print(metrics.f1_score(f.label, f.pred))
