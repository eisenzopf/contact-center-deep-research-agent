from setuptools import setup, find_packages

setup(
    name="contact_center_analysis",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "google-generativeai",
        "asyncio"
    ],
    extras_require={
        'test': [
            'pytest',
            'pytest-asyncio'
        ]
    },
    python_requires='>=3.8'
) 