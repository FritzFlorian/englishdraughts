from setuptools import setup

setup(name='englishdraughts',
      version='0.1',
      description='AI that plays english draughts using AlphaZero',
      url='https://github.com/FritzFlorian/englishdraughts',
      author='Fritz Florian',
      license='MIT',
      packages=['englishdraughts'],
      install_requires=[
          'tensorflow',
          'hometrainer'
      ],
      zip_safe=False)
