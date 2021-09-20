from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from typing import ClassVar
from sklearn.metrics import roc_curve

from .FEATTest import FEATTest

# TODO: refactor and error handling
@dataclass
class SubgroupMetricThreshold(FEATTest):
    '''
    Test if at the current probability thresholds, for a particular attribute, the fpr/tpr of its groups 
    passes the maximum/mininum specified metric thresholds. Output the list of groups which fails the test. 

    :attr: protected attribute
    :metric: choose from ['fpr','tpr']
    :metric_threshold: To pass, fpr has to be lower than the threshold or tpr has to be greater than the thresholds specified
    :proba_thresholds: optional argument. dictionary object with keys as the attribute groups and the values as the thresholds 
                       for the output to be classified as 1, default input will set thresholds of each group to be 0.5
    '''

    attr: str
    metric: str
    metric_threshold: float
    proba_thresholds: dict = None

    technique: ClassVar[str] = 'Subgroup Metric Threshold'


    def get_result(self, df_test_with_output) -> any:
        '''
        Test if at the current probability thresholds, for a particular attribute, the fpr/tpr of its groups 
        passes the maximum/mininum specified metric thresholds. Output the list of groups which fails the test.
        '''
        result = []
        self.fpr = {}
        self.tpr = {}
        self.thresholds_lst = {}
        self.thresholds = {}
        self.idx = {}

        for value in df_test_with_output[self.attr].unique():
            output_sub = df_test_with_output[df_test_with_output[self.attr] == value]
            fpr, tpr, thresholds_lst = roc_curve(output_sub['truth'],  output_sub['prediction_probas'])

            if self.proba_thresholds and isinstance(self.proba_thresholds, dict):
                threshold = self.proba_thresholds[value]
            else:
                # if threshold dict is not specified, show the markers for default probability threshold = 0.5
                threshold = 0.5

            tmp = [i for i in thresholds_lst - threshold if i > 0]
            idx = tmp.index(tmp[-1])

            self.fpr[value] = fpr
            self.tpr[value] = tpr
            self.thresholds_lst[value] = thresholds_lst
            self.thresholds[value] = threshold
            self.idx[value] = idx

            crossed_fpr_threshold = self.metric == 'fpr' and fpr[idx] > self.metric_threshold
            crossed_tpr_threshold = self.metric == 'tpr' and tpr[idx] < self.metric_threshold

            if crossed_fpr_threshold or crossed_tpr_threshold:
                result.append(value)

        return result


    def plot(self):
        ''' Plots an ROC curve for every value that crossed the metric threshold. '''
        if not self.result:
            raise AttributeError('Cannot plot before obtaining results.')
        
        colors = ['red', 'blue', 'grey', 'green', 'black', 'brown', 'purple', 'orange', 'magenta', 'pink']
        
        for value in self.results:
            color = colors.pop(0)
            tpr = self.tpr[value]
            idx = self.idx[value]
            fpr = self.fpr[value]
            thresholds_lst = self.thresholds_lst[value]
            threshold = self.thresholds[value]

            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds_lst[optimal_idx + 1]
            optimal_txt='Optimal Prob Threshold'
            txt='Current Prob Threshold'
            
            plt.scatter(
                fpr[optimal_idx],
                tpr[optimal_idx],
                color=color, 
                marker ='.',
                s=70, 
                label=f'{optimal_txt} = {str(optimal_threshold)}, {self.attr}_{value}'
            )

            plt.scatter(
                fpr[idx],
                tpr[idx],
                color=color, 
                marker ='x',
                s=30, 
                label=f'{txt} = {str(threshold)}, {self.attr}_{value}'
            )

            plt.plot(fpr, tpr, label=f'ROC of {self.attr}_{value}', color=color)
        
        plt.xlabel('False Positive Rate', fontsize=15)
        plt.ylabel('True Positive Rate', fontsize=15)

        if self.metric == 'tpr':
            plt.axhline(
                y=self.metric_threshold, 
                color='black', 
                linestyle='--', 
                label=f'Mininum TPR Threshold = {str(self.metric_threshold)}'
            )
        elif self.metric == 'fpr':
            plt.axvline(
                x=self.metric_threshold, 
                color='black', 
                linestyle='--', 
                label=f'Maximum FPR Threshold = {str(self.metric_threshold)}'
            )

        plt.title(f'ROC Curve of {self.attr} groups', fontsize=15)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()


    def run(self, df_test_with_output) -> bool:
        '''
        Runs test by calculating result and evaluating if it passes a defined condition. 

        :df_test_with_output: evaluation set dataframe containing protected attributes with 'prediction_probas' and 'truth' columns,
                            protected attribute should not be encoded yet
        '''
        self.result = self.get_result(df_test_with_output)   
        self.passed = False if self.result else True

        return self.passed