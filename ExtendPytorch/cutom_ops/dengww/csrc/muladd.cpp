#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

extern "C"
{
  PyObject *PyInit__C(void)
  {
    /**
     * 创建一个可以从 Python 导入的虚拟空 _C 模块。
     * 从 Python 导入将加载此扩展中由此文件组成的 .so，
     * 以便运行下面的 TORCH_LIBRARY 静态初始化程序。
     */
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_C", // 模块名称
        NULL, // 模块文档
        -1,   // 模块每个解释器状态的大小，或者如果模块将状态保存在全局变量中，则为 -1。
        NULL,
    };
    return PyModule_Create(&module_def);
  }
}

namespace dengww
{

  at::Tensor mymuladd_cpu(const at::Tensor &a, const at::Tensor &b, double c)
  {
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
    const float *a_ptr = a_contig.data_ptr<float>();
    const float *b_ptr = b_contig.data_ptr<float>();
    float *result_ptr = result.data_ptr<float>();
    for (int64_t i = 0; i < result.numel(); i++)
    {
      result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
    }
    return result;
  }

  at::Tensor mymul_cpu(const at::Tensor &a, const at::Tensor &b)
  {
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
    const float *a_ptr = a_contig.data_ptr<float>();
    const float *b_ptr = b_contig.data_ptr<float>();
    float *result_ptr = result.data_ptr<float>();
    for (int64_t i = 0; i < result.numel(); i++)
    {
      result_ptr[i] = a_ptr[i] * b_ptr[i];
    }
    return result;
  }

  // 改变其某个输入的运算符的示例
  void myadd_out_cpu(const at::Tensor &a, const at::Tensor &b, at::Tensor &out)
  {
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(b.sizes() == out.sizes());
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);
    TORCH_CHECK(out.dtype() == at::kFloat);
    TORCH_CHECK(out.is_contiguous());
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CPU);
    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    const float *a_ptr = a_contig.data_ptr<float>();
    const float *b_ptr = b_contig.data_ptr<float>();
    float *result_ptr = out.data_ptr<float>();
    for (int64_t i = 0; i < out.numel(); i++)
    {
      result_ptr[i] = a_ptr[i] + b_ptr[i];
    }
  }

  // 定义算子
  TORCH_LIBRARY(dengww, m)
  {
    m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
    m.def("mymul(Tensor a, Tensor b) -> Tensor");
    m.def("myadd_out(Tensor a, Tensor b, Tensor(a!) out) -> ()");
  }

  // 为上面定义的算子注册 CPU 实现
  TORCH_LIBRARY_IMPL(dengww, CPU, m)
  {
    m.impl("mymuladd", &mymuladd_cpu);
    m.impl("mymul", &mymul_cpu);
    m.impl("myadd_out", &myadd_out_cpu);
  }
}