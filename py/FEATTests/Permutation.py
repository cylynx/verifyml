from dataclasses import dataclass
from pandas import DataFrame, Series
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import ClassVar

from .FEATTest import FEATTest

# TODO: refactor and error handling
@dataclass
class Permutation(FEATTest):
    '''
    Check if the difference/ratio of specified bias metric of any group within a specified protected attribute between 
    the original dataset and the perturb dataset exceeds the threshold. Output list of groups that fails the test.
    i.e. Flag male gender group if 
    abs(False positive rate of male group in original test data - False positive rate of male group in perturbed gender test data) 
    >threshold     
    For ratio, take the higher value as the numerator.
     
    :attr: protected attribute specified
    :metric: type of bias metric for the test, choose from ('fpr', 'fnr', 'sr'), 
             'fpr' - false positive rate, 'fnr' - false negative rate, 'sr': selection rate
    :method: type of method for the test, choose from ('diff', 'ratio')
    :threshold: threshold for difference/ratio of the metric 
    '''

    attr: str
    metric: str
    method: str
    threshold: float

    technique: ClassVar[str] = 'Permutation'


    @staticmethod
    def add_predictions_to_df(df: DataFrame, model, encoder):
        ''' Add a column to a given df with values predicted by a given model. '''
        y_pred = model.predict(df)
        df = encoder.inverse_transform(df)
        df['prediction'] = y_pred

        return df


    @staticmethod
    def get_metric_dict(
        attr: str,
        metric: str,
        df: DataFrame
    ) -> dict[str, float]:
        ''' Calculate metric differences for a protected attribute on a given df. '''
        metric_dict = {}

        for i in df[attr].unique():
            tmp = df[df[attr] == i]
            cm = confusion_matrix(tmp.truth, tmp.prediction)

            if metric == 'fpr':
                metric_dict[i] = cm[0][1]/cm[0].sum()
            elif metric == 'fnr':
                metric_dict[i] = cm[1][0]/cm[1].sum()
            elif metric == 'sr':
                metric_dict[i]=cm[1].sum()/cm.sum()
        
        return metric_dict


    @staticmethod
    def perturb_df(
        attr: str, 
        df: DataFrame,
        encoder
    ):
        ''' Perturb the protected attribute column values of a given df. '''
        df[attr] = np.random.permutation(df[attr].values)
        df = encoder.transform(df)

        return df
    

    def get_metric_dict_original(self, x_test: DataFrame, y_test: Series, model, encoder):
        ''' Get metric dict for original dataset. '''
        df_original = self.add_predictions_to_df(x_test, model, encoder)
        df_original['truth'] = y_test

        self.metric_dict_original = self.get_metric_dict(self.attr, self.metric, df_original)

        return self.metric_dict_original


    def get_metric_dict_perturbed(self, x_test: DataFrame, y_test: Series, model, encoder):
        ''' Get metric dict for perturbed dataset. '''
        df_perturbed = encoder.inverse_transform(x_test)
        df_perturbed = self.perturb_df(self.attr, df_perturbed, encoder)
        df_perturbed = self.add_predictions_to_df(df_perturbed, model, encoder)
        df_perturbed['truth'] = y_test

        self.metric_dict_perturbed = self.get_metric_dict(self.attr, self.metric, df_perturbed)

        return self.metric_dict_perturbed


    def get_result(
        self,
        x_test: DataFrame,
        y_test: Series,
        model,
        encoder
    ) -> list:
        '''
        Calculate test result. Compare the original vs perturbed metric
        dicts and output the attribute groups that failed the test.
        '''
        md_original = self.get_metric_dict_original(x_test, y_test, model, encoder)
        md_perturbed = self.get_metric_dict_perturbed(x_test, y_test, model, encoder)

        result = []

        for i in md_original.keys():
            if md_original[i] > md_perturbed[i]:
                if self.method == 'ratio':
                    val = md_original[i] / md_perturbed[i]
                elif self.method == 'diff':
                    val = md_original[i] - md_perturbed[i]
            else:
                if self.method == 'ratio':
                    val = md_perturbed[i] / md_original[i]
                elif self.method == 'diff':
                    val = md_perturbed[i] - md_original[i]

            if val > self.threshold :
                result.append(i)

        return result


    def run(
        self,
        x_test: DataFrame,
        y_test: Series,
        model,
        encoder,
    ) -> bool:
        '''
        Runs test by calculating result and evaluating if it passes a defined condition. 

        :x_test: dataframe containing features to be inputted into the model predictions
        :y_test: array/list/series containing the truth of x_test
        :model: model object
        :encoder: one hot encoder object, to allow for permutation of the protected attribute
        '''
        self.result = self.get_result(x_test, y_test, model, encoder)
        self.passed = False if self.result else True

        return self.passed