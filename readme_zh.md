# Moon Location 安装与运行指南
## 一、Installation（安装）
### 1. 创建并激活conda环境
```bash
conda create -n moon_location
conda activate moon_location
```

### 2. 安装依赖库
```bash
pip install -r requirements.txt
```
- torch 是运行匹配算法的核心依赖。
- 可通过 PyTorch 官网（https://pytorch.org/）下载适配版本。

---

## 二、Running（运行）
### 方式一：直接启动软件界面
```bash
python main.py
```

### 方式二：分步运行获取中间结果
1. 按顺序运行 utils 目录下的相关代码。
2. 详细步骤可参考 `Threadclass.py` 中的说明。


要不要我帮你补充**环境配置常见问题排查清单**，或优化命令行代码的格式排版？


