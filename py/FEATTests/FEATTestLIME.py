from dataclasses import dataclass
from typing import ClassVar

from FEATTests import FEATTest

@dataclass
class FEATTestLIME(FEATTest):
    """ A FEAT test using the LIME method """
    
    # df: DataFrame
    col1: str
    col2: str
    technique: ClassVar[str] = 'LIME'
    technique_desc: ClassVar[str] = ' '.join('''
        Local surrogate models are interpretable models that are used to
        explain individual predictions of black box machine learning models.
        
        Local interpretable model-agnostic explanations (LIME) is a paper
        in which the authors propose a concrete implementation of local
        surrogate models.
    '''.split(None))


    def run(self) -> bool:
        # LIME-specific logic here
        self.result = self.col1
        self.passed = self.result == self.col1 + self.col2

        return self.passed