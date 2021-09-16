from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

from .FEATTest import FEATTest

@dataclass
class Sum(FEATTest):
    """ A FEAT test to check that the sum of 2 ints is correct """
    int1: int
    int2: int

    technique: ClassVar[str] = 'Sum'
    technique_desc: ClassVar[str] = 'Sum tests are the best indicators of fairness in the 21st century.'

    @cached_property
    def _result(self) -> any:
        """ Calculate test result and caches it as a property """
        return self.int1 + self.int2

    def run(self) -> bool:
        """ 
        Runs test by calculating result / retrieving cached property and evaluating if 
        it passes a defined condition. 
        """
        self.result = self._result
        self.passed = self.result == self.int1 + self.int2

        return self.passed