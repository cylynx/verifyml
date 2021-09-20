from dataclasses import dataclass, field
from pandas import DataFrame
from typing import ClassVar

from .FEATTest import FEATTest

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

    technique: ClassVar[str] = 'Data Shift'


    def get_df_distribution_by_pa(self, df: DataFrame, col: str):
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