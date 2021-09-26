from abc import ABC, abstractmethod
from dataclasses import dataclass, field

@dataclass
class ModelTest(ABC):
    """Base class for all FEAT tests"""
    
    test_name: str = field(init=False, default=None)
    test_desc: str = field(init=False, default=None)
    result: any = field(init=False, default=None)
    passed: bool = field(init=False, default=None)

    @abstractmethod
    def run(self) -> bool:
        """
        Contains logic specific to a model test. This method should:
        1) Get test result and update self.result
        2) Update self.passed, a boolean indicating if result meets a defined condition
        """
