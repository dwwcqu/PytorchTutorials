# C++/CUDA Extensions in PyTorch

为 PyTorch 编写 C++/CUDA 扩展的示例。请参阅[此处](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html)获取随附的教程。此 repo 演示了如何编写具有自定义 CPU 和 CUDA 内核的示例 `extension_cpp.ops.mymuladd` 自定义操作。

此 repo 中的示例适用于 PyTorch 2.4+。

## 编译

```bash
$ pip install --no-build-isolation -e .
```

## 测试

```bash
$ python test/test_dengww.py
```