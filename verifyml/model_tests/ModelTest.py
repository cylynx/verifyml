# Copyright 2021 Cylynx
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ModelTest(ABC):
    """Base class for all FEAT tests"""

    result: any = field(init=False, default=None)
    passed: bool = field(init=False, default=None)

    @abstractmethod
    def run(self) -> bool:
        """
        Contains logic specific to a model test. This method should: 
        1) Get test result and update self.result 
        2) Update self.passed, a boolean indicating if result meets a defined condition
        """
