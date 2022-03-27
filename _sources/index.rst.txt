.. translator documentation master file, created by
   sphinx-quickstart on Thu Nov 19 13:30:32 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

tkitAutoMask文档
======================================

自动构建掩码 加入多种动态掩码合集，上下三角和动态片段，以及默认的概率

-上三角，实现类似从左到右的预测，就是单向注意，用于续写,以及unilm_mask的注意力矩阵。

片段，连续多个mask，更加适合解决补全。
未来尝试加入 模板预测掩码

安装

>>> pip install tkitAutoMask

https://github.com/napoler/tkit-automask



.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   res/modules
   


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

作者博客
https://www.terrychan.org/
