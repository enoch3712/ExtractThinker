from setuptools import setup

setup(
    name="extract_thinker_eval",
    entry_points={
        "console_scripts": [
            "extract_thinker-eval=extract_thinker.eval.cli:main",
        ],
    },
) 