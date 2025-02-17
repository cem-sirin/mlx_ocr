from setuptools import setup, find_packages

setup(
    name="mlx-ocr",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,  # Include package data files
    package_data={
        "mlx_ocr": [
            "misc/vocabs/*.txt",
            "misc/fonts/*.ttf",
        ]
    },  # Include txt files in mlx_ocr/misc/vocabs
    install_requires=["mlx>=0.0.5", "numpy>=1.21.0", "pillow>=8.0.0"],
    author="Cem Sirin",
    description="An OCR library implemented in MLX",
    license="MIT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cem-sirin/mlx-ocr",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS",
    ],
    python_requires=">=3.8",
    keywords="OCR, MLX, deep learning, machine learning, apple-silicon",
)
