import gradio as gr
import whisper

print("Loading Whisper tiny model (this may take a moment on first run)...")
model = whisper.load_model("tiny")

def transcribe(audio):
    if audio is None:
        return "No audio provided."
    print(f"Processing audio: {audio}")
    result = model.transcribe(audio)
    return result["text"]

demo = gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(type="filepath"), 
    outputs="text",
    title="Local Whisper Transcription",
    description="Upload an audio file to transcribe it locally using OpenAI's Whisper model."
)

if __name__ == "__main__":
    demo.launch()
