"""Prints the Triton requirement that the installed torch declares on PyPI."""

import json
import urllib.request

import torch

version = torch.__version__.split("+")[0]
url = f"https://pypi.org/pypi/torch/{version}/json"
with urllib.request.urlopen(url, timeout=30) as response:
    requires = json.load(response)["info"]["requires_dist"]
print(next(r.split(";")[0].strip() for r in requires if r.startswith("triton")))
