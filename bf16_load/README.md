这是个有趣的发现（对于大佬可能是基操），当从bfloat16*的地址加载数据并进行类型转换时，即加载到float*上，可以根据bf16的结构特性，即（fp32: E8M23，fp16: E5M10， bf16: E8M7），所以可以通过mask间隔着读取，然后低位补0（即后23-7位补0）即可优美的进行：
```
static __device__ inline void vload2_lm_unordered_infer(
        const bfloat16* ptr,
        float32x16_t& veven,
        float32x16_t& vodd) {
    constexpr int mask = 0xaaaaaaaa;  // 0b10101010101010101010101010101010
    veven = reinterpret_cast<float32x16_t>(vload_lm_int16x32_mz(ptr, mask));
    vodd = reinterpret_cast<float32x16_t>(vshuffle2_float16x32(
            reinterpret_cast<float16x32_t>(vload_lm_int16x32_mz(ptr, (~mask)))));
}
```

这里给出对reinterpret_cast的理解：
```
#include <iostream>
#include <string>

class Base {
public:
    virtual void speak() { std::cout << "I'm Base\n"; }
};

class Derived : public Base {
public:
    void speak() override { std::cout << "I'm Derived\n"; }
};

int main() {
    // --- static_cast ---
    std::cout << "--- static_cast ---" << std::endl;
    int i = 42;
    double d = static_cast<double>(i); // int -> double
    std::cout << "int to double: " << d << std::endl;

    // --- const_cast ---
    std::cout << "\n--- const_cast ---" << std::endl;
    const std::string str = "Hello";
    std::string& non_const_str = const_cast<std::string&>(str);
    non_const_str[0] = 'h'; // 修改const对象（注意：这是未定义行为，但演示用）
    std::cout << "Modified string: " << str << std::endl;

    // --- dynamic_cast ---
    std::cout << "\n--- dynamic_cast ---" << std::endl;
    Base* base = new Derived();
    Derived* derived = dynamic_cast<Derived*>(base); // Base* -> Derived*
    if (derived) {
        derived->speak(); // 成功转型，调用Derived的speak
    } else {
        std::cout << "dynamic_cast failed\n";
    }
    delete base;

    // --- reinterpret_cast ---
    std::cout << "\n--- reinterpret_cast ---" << std::endl;
    int x = 0x3F800000; // float 1.0的bit表示
    float* fx = reinterpret_cast<float*>(&x);
    std::cout << "Interpreted int as float: " << *fx << std::endl;

    return 0;
}
```
