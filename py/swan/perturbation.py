import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

def perturbation(
    protected_attr: list,
    test,
    y_test,
    output,
    ens,
    estimator
):
    for pa in protected_attr:
        fnr = {}
        fpr = {}
        rr={}
        test_perturb=test.copy()
        test_perturb[pa]=np.random.permutation(test_perturb[pa].values)
        test_perturb=ens.transform(test_perturb)
        y_pred = estimator.predict(test_perturb)
        output_perturb=ens.inverse_transform(test_perturb)
        output_perturb['truth']=y_test
        output_perturb['prediction']=y_pred
        for i in output[pa].unique():
            print(pa+' '+i)
            tmp=output_perturb[output_perturb[pa] == i]
            cm=confusion_matrix(tmp.truth, tmp.prediction)
            print(cm)
            print('\n')
            precision, recall, fscore, support = score(tmp.truth,tmp.prediction)
            print('precision: {}'.format(precision))
            print('recall: {}'.format(recall))
            print('fscore: {}'.format(fscore))
            print('support: {}'.format(support))
            print('\n')
            fnr[i] = 1-recall[1]
            fpr[i] = 1-recall[0]
            rr[i]=cm[1].sum()/cm.sum()
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
        axs[0].bar(list(fnr.keys()), list(fnr.values()))
        axs[1].bar(list(fpr.keys()), list(fpr.values()))
        axs[2].bar(list(rr.keys()), list(rr.values()))
        fig.suptitle('False Negative rates and False Positive rates, Rejection rates across '+ pa)