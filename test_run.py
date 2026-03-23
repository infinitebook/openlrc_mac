import os
from openlrc import LRCer
from openlrc.config import TranscriptionConfig

def main():
    # 刚才我们修改了 config.py 的默认值，如果你不传参数，默认就会使用刚才配置好的 whisper.cpp 路径
    transcription_config = TranscriptionConfig()
    
    # 实例化打轴器
    lrcer = LRCer(transcription=transcription_config)
    
    # 测试音频路径：项目自带了一个测试音频
    test_audio = "tests/data/test_audio.wav"
    
    if not os.path.exists(test_audio):
        print(f"❌ 找不到测试音频: {test_audio}")
        return

    print("🚀 开始测试 Whisper.cpp 纯打轴流 (跳过 LLM 翻译)...")
    
    # 这里的关键是 skip_trans=True，这样就不会触发需要 API Key 的大模型翻译步骤
    # 专心验证我们魔改的底层 C++ 引擎是否桥接成功
    lrcer.run(test_audio, skip_trans=True)
    
    print("✅ 测试执行完毕！请检查 tests/data/ 目录下是否成功生成了字幕文件。")

if __name__ == "__main__":
    main()
