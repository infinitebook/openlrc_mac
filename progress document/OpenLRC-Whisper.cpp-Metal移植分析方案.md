# OpenLRC 到 Whisper.cpp + Metal 的移植分析方案

## 1. 当前项目架构总结 (基于 faster-whisper + CUDA)

OpenLRC 的核心处理流程和模块分工如下：

- **核心调度层 (`openlrc/openlrc.py`)**：通过 `LRCer` 类统筹整个处理流程。`run()` 方法采用 Producer-Consumer 模式，将音频转录 (Transcription) 和字幕翻译 (Translation) 解耦并并发处理。
- **预处理模块 (`openlrc/preprocess.py`)**：
  - 提取视频的音频流并转为 `.wav` 格式 (`ffmpeg`)。
  - 使用 `ffmpeg-normalize` 进行响度归一化 (Loudness Normalization)。
  - (可选) 使用 `deepfilternet` (`df.enhance`) 进行基于深度学习的降噪 (Noise Suppression)。
- **转录模块 (`openlrc/transcribe.py`)**：
  - 依赖 `faster-whisper` 的 `WhisperModel` 和 `BatchedInferencePipeline` 进行加速推理。
  - 内置了 Silero VAD 过滤以跳过无声片段。
  - **关键依赖**：`faster-whisper` 输出的 `Segment` 对象中包含单词级别的打点信息 (Word-level timestamps)。这些内容被 `openlrc` 的 `sentence_split` （借助 `pysbd` 和 `spacy`）方法用来重新切分并组成具有合理自然断句的字幕。
- **翻译与优化层 (`openlrc/translate.py` & `openlrc/opt.py`)**：调用 LLM (OpenAI/Anthropic) 进行批量上下文感知的翻译，然后使用 `SubtitleOptimizer` 扩展字幕持续时间等。

## 2. API 差异对比：`faster-whisper` vs `whisper.cpp`

要将底层的推理后端从 `faster-whisper` 切换到 `whisper.cpp` (通过 Python binding 例如 `whisper-cpp-python` 或 `pywhispercpp`)，存在以下主要差异：

### 2.1 模型加载与执行方式
- **`faster-whisper`**：下载或本地加载 CTranslate2 格式模型。
  ```python
  model = WhisperModel(model_name, device="cuda", compute_type="float16")
  pipeline = BatchedInferencePipeline(model)
  seg_gen, info = pipeline.transcribe(audio_path, vad_filter=True, word_timestamps=True)
  ```
- **`whisper.cpp`**：通常基于 `.bin` 格式 (GGML/GGUF 权重) 模型，并通过底层 C++ 库进行推理。
  ```python
  from whisper_cpp_python import Whisper
  model = Whisper(model_path="ggml-large-v3.bin", n_gpu_layers=-1) # -1 启用 Metal
  segments = model.transcribe(audio_path)
  ```

### 2.2 VAD (Voice Activity Detection) 处理
- `faster-whisper` 在 `transcribe` 接口中内嵌了 `vad_filter=True` 的支持，它会在送入模型推理之前，调用 Silero VAD 把静音段剔除并产生 `duration_after_vad` 的统计。
- `whisper.cpp` 的原生 Python binding 未必包含开箱即用的优质 VAD 逻辑。如果没有，移植时需要手动引入 `silero-vad` 来处理音频，将其切分后再交由 `whisper.cpp` 推理，并统计丢弃的静音时长 (`vad_ratio`)。

### 2.3 `Segment` 返回结构适配
在 `openlrc/transcribe.py` 中，强依赖了 `faster-whisper` 的返回属性：
- `Segment`: 属性包括 `id, seek, start, end, text, tokens, avg_logprob, words` 等。
- `Word`: 属性包括 `start, end, word, probability`。
`whisper.cpp` 返回的结构通常是个包含字典的列表，需要在代码中套一层适配器模式（Adapter Pattern）或数据转换层，将其输出结果包装成符合 `Segment` 和 `Word` 的 `NamedTuple` 或类属性集合。

## 3. Metal 加速的可行性与环境配置

Apple Silicon (M1/M2/M3) 原生支持 Metal Performance Shaders (MPS)。`whisper.cpp` 的底层已经高度优化了基于 `ggml-metal` 的计算后端，完美契合 macOS 。

**环境配置与依赖重构方案：**
1. 移除 CUDA 相关依赖 (`torch` 的 cu124 版本，`faster-whisper` 中依赖的 CTranslate2, cuBLAS 等)。
2. 引入 `whisper-cpp-python` 时，需要启用 Metal 编译标志。
   ```bash
   CMAKE_ARGS="-DWHISPER_METAL=on" pip install whisper-cpp-python
   ```
3. 在代码配置中 (`openlrc/config.py` 和 `openlrc/defaults.py`), 原有的 `device="cuda"` 应该由模型初始化时的 `n_gpu_layers=-1` 替代以触发 Metal 卸载。

## 4. 具体的代码重构步骤

### 步骤一：封装 `whisper.cpp` 接口适配器
新建一个 `openlrc/whisper_cpp_backend.py`，专门封装对 `whisper_cpp_python.Whisper` 的调用。在这里实现并模仿 `faster_whisper` 的输出结构：
- 定义匹配 `faster_whisper.transcribe` 中 `Segment` 和 `TranscriptionInfo` 接口的类。
- 实现 `Word` 的映射封装，确保 `word_timestamps` 能正确传递出来。

### 步骤二：独立 VAD 处理逻辑 (如需)
如果 `whisper-cpp-python` 无法直接通过参数过滤 VAD：
- 引入使用 `torchaudio` 或 ONNX 加载 `silero-vad` 模型。
- 在输入 `whisper.cpp` 前对音频作裁剪。
- 在结果的 `Info` 中计算 `duration` 与 `duration_after_vad`，并拼接最终的时间戳，以维持原有的 `vad_ratio` 逻辑输出。

### 步骤三：修改 `Transcriber` 核心类 (`openlrc/transcribe.py`)
- 重构 `Transcriber.__init__`，移除 `WhisperModel` 和 `BatchedInferencePipeline` 的实例化。
- 添加根据当前 OS (`darwin`) 自动拉取 GGML 格式模型或让用户指定本地 GGML 模型路径的逻辑。
- 实例化新封装的 `WhisperCppAdapter`，接收 `asr_options`。将 `device` 的判断转换为启用 Metal 的判断。

### 步骤四：清理无用配置及更新依赖
- 在 `pyproject.toml` 或 `uv` 的配置中，剔除 `faster-whisper` 及其强依赖，加入 `whisper-cpp-python` 以及可能单独需要的 `silero-vad` 库。
- 修改 `README.md` 的安装指南，提示 macOS 用户通过 `CMAKE_ARGS="-DWHISPER_METAL=on"` 来安装。

---
**结论：** 
由于 OpenLRC 在工程上已经做到了较好的模块化（单独拆出了 `openlrc/transcribe.py` 中的 `Transcriber` 类处理推理任务），将该组件的推理引擎替换为具有 Metal 支持的 `whisper.cpp` 在架构上高度可行。最大的难点是兼容 `faster-whisper` 提供的高级 VAD 过滤特性以及将其返回的词级时间戳准确适配给上层使用的 NLP 断句算法中。

## 5. 方案对比：使用 Python Binding vs 作为命令行工具调用

如果您倾向于将 `whisper.cpp` 编译后的独立二进制程序 (Executable) 直接作为命令行工具调用（即通过 Python 的 `subprocess` 模块起进程并收集结果），而不是使用 Python Binding，这会带来不同维度的影响。总体而言，这会**显著降低部署与环境配置的难度**，但在**代码集成和进程内交互的复杂度上会略微增加**。以下是具体的对比分析：

### 5.1 难度降低的方面 (优势)
1. **彻底解耦，依赖地狱消失**：
   - Python Binding在不同系统平台往往容易因为编译环境（如 Xcode CommandLine Tools, CMake 版本等）出现安装挫折，例如 `pip install whisper-cpp-python` 时无法正确启用 Metal 支持。
   - 使用可执行文件方案，意味着您只需要在系统里准备好已编译支持 Metal 的 `whisper-cli` 二进制文件。Python 端不再需要任何沉重的 C++ 构建依赖，实现了绝对的代码层解耦。
2. **规避 Python GIL 与内存安全风险**：
   - 推理过程是一个独立的进程，它拥有独立的内存空间。这意味着大模型推理时占用的大量内存会在子进程结束后被操作系统干净地回收，且不会和 Python 的垃圾回收或并发执行 (GIL) 机制产生任何干涉。
3. **更容易的分发与打包**：
   - 对于要封装给普通用户使用的 Mac 软件应用，直接将预编译好的二进制可执行文件打包附带进程序内部是最可靠的方法，避免了目标机器缺乏对应 Python 或 C 编译环境的问题。

### 5.2 难度增加的方面 (工程挑战)
1. **JSON 序列化与文件 I/O 开销**：
   - 之前在内存中直接流转的 Python 对象 (如 `Segment` )，现在需要两步转换：`whisper.cpp` 将结果输出到磁盘的文件（例如使用参数 `-oj -ml 1` 输出带有词级时间戳的 JSON），然后 `openlrc` 的 Python 代码需要去读取此磁盘文件并重新解析成内部需要的数据结构。
   - 这多了一步 I/O 开销及解析耗时（尽管对于整段大音频来说这个延时微乎其微）。
2. **VAD 机制的实现变化**：
   - `faster-whisper` 内置的 Silero VAD 可以在喂给核心模型前就把音频直接在内存里切为非静音段；而调用外部进程时，很难在中间横插一脚。
   - 如果不满意命令行的内部静音处理，那么需要在使用 `subprocess` 启动进程前，先独自加载一个 VAD 模型，通过 ffmpeg 将音频硬切成多个临时文件，再循环调用 `whisper-cli` 多次——这会带来进程频繁起停的巨大开销。
   - 如果直接信任 `whisper.cpp` 的断句能力，则可能导致无法生成完全符合 OpenLRC 原始计算预期的精确 `vad_ratio` 统计数据。
3. **实时进度回调 (Progress Callback) 更繁琐**：
   - 在 Python Binding 下，转录是通过生成器 (Generator) 一块一块返回结果的，你可以非常轻松地在 Python 里面用 `tqdm` 渲染出平滑的进度条。
   - 命令行模式下，如果想要获取实时进度回显给控制台，必须用非阻塞 (Non-blocking) 的方式持续读取并解析子进程的 `stderr`/`stdout` 管道内容，从中正则提取时间进度或百分比标志，代码编写上更加繁琐且容易在不同平台发生死锁 (Deadlock)。

### 总结
改为**“命令行调用” (Subprocess 模式)** 是一条非常典型的**工业级稳定做法**。

- **难度变化综合预判**：对开发者而言，实现上的**底层代码工作量略微增加**（需要重写进度条监听管道和 JSON 反序列化适配代码），但对于**最终用户的安装体检和程序的长期可用性来说，部署难度和运行出错几率将大幅下降**。
- **推荐策略**：如果您更看重应用跑在广大不同 Mac 用户电脑上的即插即用体验，**非常建议采用此方案**。您只需在 Python 层面实现一个类似于 `WhisperCLIBackend` 的类，统一封装 `subprocess.run/Popen` 启动命令、临时文件管理和管道监听逻辑即可。
