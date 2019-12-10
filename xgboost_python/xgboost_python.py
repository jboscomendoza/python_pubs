import requests
import matplotlib
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

url_data = "https://raw.githubusercontent.com/jboscomendoza/rpubs/xgboost_r/xgboost_en_r/agaricus-lepiota.data"

url_names = "https://raw.githubusercontent.com/jboscomendoza/rpubs/xgboost_r/xgboost_en_r/agaricus-lepiota.data"

hongos_data = requests.get(url_data)
hongos_names = requests.get(url_names)

open("agaricus-lepiota.data", "wb").write(hongos_data.content)
open("agaricus-lepiota.data", "wb").write(hongos_names.content)

nombres = [
    "target", "cap_shape", "cap_surface", "cap_color", "bruises", "odor", 
    "gill_attachment", "gill_spacing", "gill_size", "gill_color", "stalk_shape",
    "stalk_root", "stalk_surface_above_ring", "stalk_surface_below_ring", 
    "stalk_color_above_ring", "stalk_color_below_ring", "veil_type", 
    "veil_color", "ring_number", "ring_type", "spore_print_color", "population",
    "habitat"
]

hongo = pd.read_table("agaricus-lepiota.data", sep=",", names=nombres)

hongo_target = hongo["target"].replace(["e", "p"], [0, 1])
hongo_feat   = pd.get_dummies(hongo.drop("target", axis=1))

feat_train = hongo_feat.sample(frac=0.7, random_state=1919)
feat_test  = hongo_feat.drop(feat_train.index)

target_train = hongo_target.iloc[feat_train.index]
target_test  = hongo_target.drop(feat_train.index)

hongo_train_mat = xgb.DMatrix(feat_train, label=target_train)
hongo_test_mat  = xgb.DMatrix(feat_test,  label=target_test)

params = {
    "max_depth": 2, "eta": 0.3, "objective": "binary:logistic",
    "nthread":2, "verbose_eval": True
}

num_rounds = 10

evallist = [(hongo_train_mat, 'eval'), (hongo_test_mat, 'train')]
evallist = [(hongo_train_mat, 'eval')]

modelo = xgb.train(params, hongo_train_mat, num_rounds, evallist)

xgb.plot_importance(modelo)

plt.show()

prediccion = list(modelo.predict(hongo_test_mat))
prediccion = [1 if i > 0.5 else 0 for i in prediccion]

pd.DataFrame({"target": target_test, "predict":prediccion})

print(confusion_matrix(target_test, prediccion))
print(accuracy_score(target_test, prediccion))
print(classification_report(target_test, prediccion))