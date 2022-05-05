from setuptools import setup

setup(
    name='GNNSubNet',
    packages=['GNNSubNet'],
    version='0.1.0',
    author=" Bastian Pfeifer",
    author_email="damien.j.martin@gmail.com",
    description='Disease Subnetwork Detection with Explainable Graph Neural Networks',
    long_description='Disease Subnetwork Detection with Explainable Graph Neural Networks',
    license='MIT',
    url='https://github.com/pievos101/GNN-SubNet',                            # URL to GitHub repo
    # download_url='https://github.com/mdbloice/Augmentor/tarball/0.1.1',   # Get this using git tag
    keywords=['graph-neural-networks', 'disease-networks', 'explainable-ai'],
    include_package_data=True,  # This will include all files in MANIFEST.in in the package when installing.
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
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
    include_package_data=True,
    package_data={'': ['datasets/Synthetic/*.txt']}
)