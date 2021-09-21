from __future__ import annotations
from dataclasses import dataclass, field
from pandas import DataFrame
from typing import ClassVar
import matplotlib.pyplot as plt

from .FEATTest import FEATTest
from .utils import plot_to_str

@dataclass
class DataShift(FEATTest):
    '''
    Test if there is any shift (based on specified threshold) in the distribution of the protected feature, 
    which may impose new unfairness and require a retraining of the model, output the shifted attributes.
    
    :protected_attr: list of protected attributes
    :threshold: probability distribution threshold of an attribute, where if the difference between training data
                     distribution and evalaution distribution exceeds the threhold, the attribute will be flagged
    '''

    protected_attr: list[str]
    threshold: float
    plots: dict[str, str] = field(repr=False, default_factory=lambda: {})

    technique: ClassVar[str] = 'Data Shift'


    @staticmethod
    def get_df_distribution_by_pa(df: DataFrame, col: str):
        '''
        Get the probability distribution of a specified column's values in a given df.
        '''
        df_dist = df.groupby(col)[col].apply(lambda x: x.count()/len(df))

        return df_dist


    def get_result(self, df_train: DataFrame, df_eval: DataFrame) -> any:
        ''' 
        Calculate test result.

        :df_train: training data features, protected features should not be encoded yet
        :df_eval: data to be evaluated on, protected features should not be encoded yet
        '''
        _result = []

        for pa in self.protected_attr:
            train_dist = self.get_df_distribution_by_pa(df_train, pa)
            eval_dist = self.get_df_distribution_by_pa(df_eval, pa)
            
            if sum(abs(train_dist - eval_dist) > self.threshold):
                _result.append(pa)

        return _result


    def plot(self, df_train, df_eval):
        '''
        Plot the distribution of the attribute groups for training and evaluation set
        
        :df_train: training data features, protected features should not be encoded yet
        :df_eval: data to be evaluated on, protected features should not be encoded yet
        '''
        fig, axs = plt.subplots(1, len(self.protected_attr), figsize=(15, 4),)
        num=0
        for pa in self.protected_attr:
            train_dist = self.get_df_distribution_by_pa(df_train, pa).sort_values('index')
            train_dist.plot(kind='bar', color='green', ax=axs[num])
            num+=1

        training_title = 'Probability Distribution of protected attributes in training set'
        fig.suptitle(training_title)
        plt.show()
        self.plots[training_title] = plot_to_str()

        fig, axs = plt.subplots(1, len(self.protected_attr), figsize=(15, 4),)
        num=0
        for pa in self.protected_attr:
            eval_dist = self.get_df_distribution_by_pa(df_eval, pa).sort_values('index')
            eval_dist.plot(kind='bar', color='red', ax=axs[num])
            num+=1
        test_title = 'Probability Distribution of protected attributes in test set'
        fig.suptitle(test_title)
        plt.show()        
        self.plots[test_title] = plot_to_str()


    def run(self, df_train: DataFrame, df_eval: DataFrame) -> bool:
        '''
        Runs test by calculating result / retrieving cached property and evaluating if 
        it passes a defined condition. 

        :df_train: training data features, protected features should not be encoded yet
        :df_eval: data to be evaluated on, protected features should not be encoded yet
        '''
        self.result = self.get_result(df_train, df_eval)
        self.passed = False if self.result else True

        return self.passed