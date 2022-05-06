from setuptools import setup

setup(
    name='GNNSubNet',
    packages=['GNN-SubNet'],
    version='0.1.0',
    author="Bastian Pfeifer",
    author_email="bastian.pfeifer@medunigraz.at",
    description='Disease Subnetwork Detection with Explainable Graph Neural Networks',
    long_description='Disease Subnetwork Detection with Explainable Graph Neural Networks',
    license='MIT',
    url='https://github.com/pievos101/GNN-SubNet',
    keywords=['graph-neural-networks', 'disease-networks', 'explainable-ai'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: GPL-3.0',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    install_requires=[
        'torch>=1.11.0',
        'tqdm>=4.9.0',
        'numpy>=1.22.3'
    ],
    include_package_data=True,  # Alternatively add files to MANIFEST.in
    package_data={'': ['datasets/synthetic/*.txt']}
)