import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tortoise-tts",
    packages=setuptools.find_packages(),
    version="3.1.0 beta",
    author="Nymessence",
    author_email="no official email yet",
    description="A high quality multi-voice text-to-speech library (nymessence fork)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nymessence/tortoise-tts",
    project_urls={},
    scripts=[
        'scripts/tortoise_tts.py',
    ],
    include_package_data=True,
    install_requires=[
        'tqdm',
        'rotary_embedding_torch',
        'inflect',
        'progressbar',
        'einops',
        'unidecode',
        'scipy',
        'librosa',
        'transformers>=4.31.0',
        'tokenizers>=0.14.0',
        'scipy>=1.13.1'
        # 'deepspeed==0.8.3',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
