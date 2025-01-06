import gradio as gr
from groq import Groq
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq()

def transcribe_and_respond(audio):
    # 'audio' will be the path to the audio file
    with open(audio, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=("audio.wav", file.read()),
            model="distil-whisper-large-v3-en",
            response_format="verbose_json",
        )
        transcribed_text = transcription.text

    # Generate a response using the transcribed text
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": transcribed_text},
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,
    )

    output_text = completion.choices[0].message.content
    return transcribed_text, output_text

# Define Gradio interface
interface = gr.Interface(
    fn=transcribe_and_respond,
    inputs=gr.Audio(type="filepath"),  # Change type to 'filepath'
    outputs=[
    gr.Textbox(label="Spoken Details in Text Format"),
    gr.Textbox(label="Generated Output")
]
,
    title="Voice Assistant with Groq",
    description="Record your voice, transcribe it, and get an AI response!",
)

# Launch the Gradio app
interface.launch()
