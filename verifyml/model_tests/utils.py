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

from scour import scour
from io import BytesIO, StringIO
import matplotlib.pyplot as plt
import base64


def svg_encode(svg: str):
    # Ref: https://bl.ocks.org/jennyknuth/222825e315d45a738ed9d6e04c7a88d0
    # Encode an SVG string so it can be embedded into a data URL.
    enc_chars = '"%#{}<>'  # Encode these to %hex
    enc_chars_maybe = "&|[]^`;?:@="  # Add to enc_chars on exception
    svg_enc = ""
    # Translate character by character
    for c in str(svg):
        if c in enc_chars:
            if c == '"':
                svg_enc += "'"
            else:
                svg_enc += "%" + format(ord(c), "x")
        else:
            svg_enc += c
    return " ".join(svg_enc.split())  # Compact whitespace


def plot_to_str(format: str = "svg"):
    # Utility function that will export a plot to a base-64 png encoded data URI scheme.
    if format == "svg":
        img = StringIO()
        plt.savefig(img, format="svg", bbox_inches="tight")
        scour_options = scour.sanitizeOptions()
        scour_options.remove_descriptive_elements = True
        return f"data:image/svg+xml,{svg_encode(scour.scourString(img.getvalue(), options = scour_options))}"
    else:
        img = BytesIO()
        plt.savefig(img, format=format, bbox_inches="tight")
        return f'data:image/{format};base64,{base64.encodebytes(img.getvalue()).decode("utf-8")}'
