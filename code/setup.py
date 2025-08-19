# setup.py
import os
from setuptools import setup, find_packages

# Function to read the requirements file
def read_requirements(file_path="requirements.txt"):
    """Reads requirements from a file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required requirements file not found at: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

if __name__ == "__main__":

    # Read the contents of your README file for the long description
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    # Define your project's setup
    setup(
        name="rl_portfolio",  # The name pip will use
        version="0.1.0",  # Start with an initial version
        author="Finn",
        description="project for reward shaping",
        long_description=long_description,
        long_description_content_type="text/markdown",  # Important for PyPI rendering
        license="MIT",  # Or whatever license you chose

        # --- Packaging ---
        package_dir={"": "src"},  # Tells setuptools to look for packages in the 'src' directory
        packages=find_packages(where="src"),  # Automatically find all packages in 'src' (like 'ml_project')

        # --- Dependencies ---
        python_requires=">=3.10",  # Specify the minimum Python version
        install_requires=read_requirements(), # Core dependencies needed to use the package

        # --- Classifiers (Metadata for PyPI) ---
        classifiers=[
            #"License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10",
            "Operating System :: OS Independent",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],

        # --- Entry Points (Optional: Creates command-line tools) ---
        # This allows you to run 'train-project' from the command line after installing
        #entry_points={
        #    "console_scripts": [
        #        "train-project=ml_project.train:main",
        #    ],
        #},
    )