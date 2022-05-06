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
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
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
        'numpy>=1.22.3',
        'igraph>=0.9.10',
        'scipy>=1.8.0',
        'sklearn>=1.0.2',
        'pandas>=1.4.2',
        'requests>=2.25.1',
        'networkx>=2.8',
        'torch-geometric>=2.0.4',
        'matplotlib>=3.5.1'
    ],
    include_package_data=True,  # Alternatively add files to MANIFEST.in
    package_data={'': ['datasets/synthetic/*.txt']}
)