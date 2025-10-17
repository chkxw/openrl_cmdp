__TITLE__ = "openrl_cmdp"
__VERSION__ = "v0.2.1"
__DESCRIPTION__ = "CMDP-tailored Distributed Deep RL Framework based on OpenRL"
__AUTHOR__ = "OpenRL Contributors"
__EMAIL__ = "huangsy1314@163.com"
__version__ = __VERSION__

import platform

python_version_list = list(map(int, platform.python_version_tuple()))
assert python_version_list >= [
    3,
    8,
    0,
], (
    "openrl_cmdp requires Python 3.8 or newer, but your Python is"
    f" {platform.python_version()}"
)
