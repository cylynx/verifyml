from __future__ import annotations
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from pandas import DataFrame
from typing import ClassVar

from .FEATTest import FEATTest
from .utils import plot_to_str

@dataclass
class FeatureImportance(FEATTest):
    '''
    Output the protected attributes that are listed in the top specified number of the features, 
    using feature importance values inputted by the user.

    :attrs: protected attributes
    :top_n: the top n features to be specified
    '''

    attrs: list[str]
    top_n: int
    plots: dict[str, str] = field(repr=False, default_factory=lambda: {})

    technique: ClassVar[str] = 'Self-declared Feature Importance'


    def plot(self, df: DataFrame, top_n: int):
        title = 'Feature Importance Plot'

        # Plot top n important features
        plt.figure(figsize=(15,8))
        plt.bar(df.iloc[:top_n,0], df.iloc[:top_n,1])
        plt.title(title)
        plt.ylabel('Relative Importance Value')
        plt.tight_layout()
        
        self.plots[title] = plot_to_str()


    def get_result(self, df_importance) -> any:
        '''
        Output the protected attributes that are listed in the top specified number of the features, 
        using feature importance values inputted by the user.
        '''
        df_importance_sorted = df_importance.sort_values(df_importance.columns[1], ascending=False)
        top_feats = df_importance_sorted.iloc[:self.top_n, 0]
        
        result = []

        for attr in self.attrs:
            result += [feat for feat in top_feats if f'{attr}_' in feat]

        return result


    def run(self, df_importance) -> bool:
        '''
        Runs test by calculating result and evaluating if it passes a defined condition. 

        :df_importance: A dataframe with 2 columns - first column of feature names and second column of importance values
        '''
        self.result = self.get_result(df_importance)        
        self.passed = False if self.result else True

        return self.passed