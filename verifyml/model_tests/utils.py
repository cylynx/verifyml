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

from io import BytesIO
import matplotlib.pyplot as plt
import base64


def plot_to_str():
    # Utility function that will export a plot to a base-64 encoded string.
    img = BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    return base64.encodebytes(img.getvalue()).decode("utf-8")
