import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import copy
import shap
import lightgbm as lgb
import sklearn.preprocessing 
import sklearn.feature_selection
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats.stats import spearmanr, pearsonr

import joblib


def evaluate_performance(y_test, y_pred, y_prob):
    # AUROC
    auroc = metrics.roc_auc_score(y_test,y_prob)
    auroc_curve = metrics.roc_curve(y_test, y_prob)
    # AUPRC
    auprc=metrics.average_precision_score(y_test, y_prob) 
    auprc_curve=metrics.precision_recall_curve(y_test, y_prob)
    #Accuracy
    accuracy=metrics.accuracy_score(y_test,y_pred) 

    recall=metrics.recall_score(y_test, y_pred)
    precision=metrics.precision_score(y_test, y_pred)
    f1=metrics.f1_score(y_test, y_pred)
    class_report=metrics.classification_report(y_test, y_pred,target_names = ["control","case"])

    model_perf = {"auroc":auroc,"auroc_curve":auroc_curve,
                  "auprc":auprc,"auprc_curve":auprc_curve,
                  "accuracy":accuracy,
                  "recall":recall,"precision":precision,"f1":f1,
                  "class_report":class_report}
        
    return model_perf

# Output result of evaluation
#bar plot consisted of accuracy,sensitivity,specificity,auroc,f1 score,precision,recall,auprc 
def eval_output(model_perf,output_file):
    with open(output_file,'w') as f:
        f.write("AUROC=%s\tAUPRC=%s\tAccuracy=%s\tRecall=%s\tPrecision=%s\tf1_score=%s\n" %
               (model_perf["auroc"],model_perf["auprc"],model_perf["accuracy"],model_perf["recall"],model_perf["precision"],model_perf["f1"]))
        f.write("\n######NOTE#######\n")
        f.write("#According to help_documentation of sklearn.metrics.classification_report:in binary classification, recall of the positive class is also known as sensitivity; recall of the negative class is specificity#\n\n")
        f.write(model_perf["class_report"])

# Plot AUROC of model
def plot_AUROC(model_perf,output_file):
    #get AUROC,FPR,TPR and threshold
    roc_auc = model_perf["auroc"]
    fpr,tpr,threshold = model_perf["auroc_curve"]
    #plot
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUROC (area = %0.2f)' % roc_auc) 
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC of Models")
    plt.legend(loc="lower right")
    plt.savefig(output_file,format = "pdf")


# Plot AUPRC of model
def plot_AUPRC(model_perf,output_file):
    #get AUPRC,Precision,Recall and threshold
    prc_auc = model_perf["auprc"]
    precision,recall,threshold = model_perf["auprc_curve"]
    #plot
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(recall, precision, color="red",
             lw=lw,
             label='AUPRC (area = %0.2f)' % prc_auc#legend
            ) 
    plt.plot([0, 1], [1, 0], color="navy", lw=lw, linestyle='--')#diagonal line
    #x,y axis scale 
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    #axis label
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("AUPRC of Models")
    plt.legend(loc="lower right")
    plt.savefig(output_file,format = "pdf")

#Random seed
SEED = np.random.seed(2020)
############ Data Processing ###########
print("\n...... Processing Data ......\n")
dosge= pd.read_csv('/picb/lilab5/dongdanyue/dongdanyue/Dosage/LightGBM/train_data70.txt',sep = '\t',index_col ="ID")#,index_col=0
dosge_feature=dosge.drop(['unconstrained'],axis=1)


out_dosge=dosge['unconstrained']
out_dosge.fillna(0,inplace = True)




#Input
X = copy.deepcopy(dosge_feature)
y = copy.deepcopy(out_dosge)
#TrainingSet : TestSet = 4 : 1
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = SEED)


X_train['Unique_KEGG_type'] = X_train['Unique_KEGG_type'].astype('category')
X_test['Unique_KEGG_type'] = X_test['Unique_KEGG_type'].astype('category')

print("\n...... Finished Data Processing ......\n")
########## Model Construction ##########
print("\n...... Training Model ......\n")

# LightGBM params
param_dict = {
    "learning_rate":[0.1, 0.05, 0.02, 0.015, 0.01],
    "num_leaves": range(10,36,5),# Maximum tree leaves for base learners.
    "max_depth" : [-1,2,3,4,5,10,20,25],#Maximum tree depth for base learners, <=0 means no limit.
    "min_data_in_leaf": range(1, 45, 2),#
    "feature_fraction" : [i / 10 for i in range(2,11)],#
    "metric" : ["binary_logloss"],#
    "early_stopping_rounds" : [None],#
    "n_jobs" : [-1],#
    "silent" : [True],#
    "verbose" : [-1],#
    "n_estimators" : range(50,1000,50),#
    "bagging_fraction" :  [i / 10 for i in range(2, 11)],#
    "bagging_freq" : [0, 1, 2],#
    "lambda_l1" : [0, 0.001, 0.005, 0.01,0.1,1,10],#
    "lambda_l2" : [0, 0.001, 0.005, 0.01,0.1,1,10],#
    "objective":["binary"],#
    "random_state":[2020],#
    "is_unbalance":[True]
}#参数列表

#Initiate model
model = lgb.LGBMClassifier()
#Adjust hyper-parameters with 5-fold cross validation
rscv = RandomizedSearchCV(model,#
                          param_dict,#
                          n_iter=100,# Number of parameter settings that are sampled. n_iter tradesoff runtime vs quality of the solution.
                          cv = 5,# Determines the cross-validation splitting strategy.
                          verbose = 0,# Controls the verbosity: the higher, the more messages.
                          scoring = "roc_auc",#
                          n_jobs =-1#
                         )#RandomizedSearchCV objext
gbm=rscv.fit(X_train, y_train,categorical_feature=['Unique_KEGG_type'])#model 
########## Model Evaluation ##########
print("\n...... Evaluating Model ......\n")

#Get best model with score [max(mean(auc(5 cross validation)))]
best_model = rscv.best_estimator_
#Get predict_class(y_pred) and predict_probality_for_case(y_prob) of TestSet
y_pred = best_model.predict(X_test)
#print(y_pred)
y_prob = best_model.predict_proba(X_test)[:,1]
y_prob#probability of prediction as positive,len:40(test set)

#Draw the learning Curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
train_sizes, train_scores, CV_scores=learning_curve(best_model,X_train,y_train,cv=cv,train_sizes=np.linspace(.1, 1.0, 8),scoring='neg_mean_squared_error',n_jobs=6)


y_pred = best_model.predict(X_test)
#print(y_pred)
y_prob = best_model.predict_proba(X_test)[:,1]
y_prob#probability of prediction as positive,len:40(test set)


#Get model performance
model_perf = evaluate_performance(y_test,y_pred,y_prob)
#Output result of evaluation
eval_output(model_perf,"/picb/lilab5/dongdanyue/dongdanyue/Dosage/LightGBM/output/dosage_binerary.txt")
#You can make bar plot consisted of accuracy,sensitivity,specificity,auroc,f1 score,precision,recall,auprc according to the "Evaluate_Result_TestSet.txt"
#Plot
# plot AUROC AUPRC
plot_AUROC(model_perf,"/picb/lilab5/dongdanyue/dongdanyue/Dosage/LightGBM/output/dosage__AUROC_TestSet_binerary.pdf")
plot_AUPRC(model_perf,"/picb/lilab5/dongdanyue/dongdanyue/Dosage/LightGBM/output/dosage_AUPRC_TestSet_binerary.pdf")

print("\n...... Finished Model Evaluation ......\n")
######################################

shap.summary_plot(shap_values, X_train,plot_type="bar",max_display = 40)
