# 公共 C/C++ 工具库

`2_utils/` 收集可复用于模型移植的基础工具，保留了来自 RKNN Model Zoo 的文件、音频和图像处理实现，同时提供仓库当前使用的轻量 WAV 读取器。

## 组件

| 目标 | 默认 | 依赖 | 用途 |
|---|---:|---|---|
| `fileutils` | 开启 | C 标准库 | 文件加载、读写和文本行处理 |
| `wavutils` | 开启 | C++ 标准库 | 读取 16 kHz PCM16 WAV，并将多声道平均为单声道 |
| `audioutils` | 关闭 | libsndfile | 音频读取、保存、重采样和声道转换 |
| `imageutils` | 关闭 | stb、JPEG，可选 RGA | 图像读取、格式转换和 letterbox |
| `imagedrawing` | 关闭 | 无额外库 | 图像绘制和 ASCII 字体渲染 |

默认配置只构建不依赖外部库的 `fileutils` 和 `wavutils`，不会因为没有图像或音频开发库而阻塞模型工程。

## 构建

```bash
cmake -S 2_utils -B /tmp/sophon-utils
cmake --build /tmp/sophon-utils
```

启用 libsndfile：

```bash
cmake -S 2_utils -B /tmp/sophon-utils \
  -DSOPHON_UTILS_BUILD_AUDIO=ON \
  -DSOPHON_SNDFILE_INCLUDE_DIR=/path/to/include \
  -DSOPHON_SNDFILE_LIBRARY=/path/to/libsndfile.so
```

图像工具同样通过 CMake 变量显式传入依赖路径。模型工程应优先直接链接目标库，避免复制工具源文件；当前 Zipformer 使用 `wavutils` 对应的 `wav.cpp/h` 接口。

## 复用注意

- `fileutils` 提供二进制加载、完整文件读写和动态长度文本行读取；调用方负责释放返回的内存。
- `audio_utils` 的音频缓冲区采用交错存储；重采样逐声道处理并更新 `sample_rate`，声道转换会平均全部输入声道；调用方负责释放 `audio_buffer_t.data`。
- `image_utils` 和 `image_drawing` 面向 RKNN/RGA 兼容场景，使用前需要启用对应依赖并确认目标平台的像素格式。
- 这些工具不包含模型、板卡运行时或 Sophon SDK 逻辑，算法目录可以按需选择组件。
