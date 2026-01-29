"""Setup script for RLHF Precision Forensics."""

from setuptools import setup, find_packages

setup(
    name="rlhf-precision-forensics",
    version="0.1.0",
    description="Systematic characterization of floating-point precision failure modes in RLHF pipelines",
    author="RLHF Precision Team",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.10",
    install_requires=[
        # NOTE: torch is NOT listed here - use container's version
        "transformers>=4.36.0",
        "datasets>=2.16.0",
        "accelerate>=0.26.0",
        "peft>=0.7.0",
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "plotly>=5.18.0",
        "tqdm>=4.66.0",
        "huggingface_hub>=0.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "static-kl-probe=scripts.static_kl_probe:main",
            "run-ppo-sweep=scripts.run_ppo_sweep:main",
            "run-full-sweep=scripts.run_full_sweep:main",
            "analyze-results=scripts.analyze_results:main",
        ],
    },
)
