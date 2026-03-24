import os
from openlrc import LRCer
from openlrc.config import TranscriptionConfig, TranslationConfig

def main():
    # 配置 Whisper.cpp 的底层路径（已设为默认）
    transcription_config = TranscriptionConfig()
    
    # 根据 README 的说明，由于你设置了 Google Key，我们需要指定 chatbot_model 为 google 的模型
    # 例如 gemini-1.5-flash 速度快且便宜，非常适合做双语字幕
    translation_config = TranslationConfig(
        chatbot_model="gemini-2.0-flash"
    )
    
    # 实例化打轴器，把翻译配置也传进去
    lrcer = LRCer(transcription=transcription_config, translation=translation_config)
    
    # 测试音频路径
    test_audio = "tests/data/test_audio.wav"
    
    if not os.path.exists(test_audio):
        print(f"❌ 找不到测试音频: {test_audio}")
        return

    print("🚀 开始测试 OpenLRC 终极全流程 (底层 Whisper.cpp 识别 + 顶层 Gemini 翻译)...")
    
    # 取消 skip_trans，并且通过 target_lang 指定想翻译成的目标语言（默认是 zh）
    lrcer.run(test_audio, target_lang="zh")
    
    print("✅ 全流程执行完毕！请检查 tests/data/ 目录下是否成功生成了完美的双语字幕文件。")

if __name__ == "__main__":
    main()
