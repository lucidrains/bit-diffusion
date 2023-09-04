from setuptools import setup, find_packages

setup(
  name = 'bit-diffusion',
  packages = find_packages(exclude=[]),
  version = '0.1.1',
  license='MIT',
  description = 'Bit Diffusion - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/bit-diffusion',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'denoising diffusion'
  ],
  install_requires=[
    'accelerate',
    'einops',
    'ema-pytorch',
    'pillow',
    'torch>=1.12.0',
    'torchvision',
    'tqdm'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
