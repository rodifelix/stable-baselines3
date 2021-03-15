from setuptools import setup

setup(name='pg_baseline',
      version='0.0.1',
      install_requires=['stable-baselines3==0.10.0', 'gym', 'numpy', 'torch', 'cloudpickle', 'pandas', 'matplotlib', 'torchvision']
)