import os
import io
import numpy as np
import requests
from scipy.io import wavfile
import tempfile

def transcribe_audio_openai(audio_data, sample_rate):
    """Transcribe audio using OpenAI's Whisper API instead of local model."""

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("No OpenAI API key found, falling back to local transcription")
        return None

    try:
        # Convert numpy array to WAV bytes
        audio_np = np.array(audio_data).astype(np.float32)

        # Normalize audio to 16-bit PCM
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_np * 32767).astype(np.int16)

        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            wavfile.write(tmp_file.name, sample_rate, audio_int16)
            tmp_path = tmp_file.name

        # Send to OpenAI Whisper API
        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        with open(tmp_path, 'rb') as audio_file:
            files = {
                'file': ('audio.wav', audio_file, 'audio/wav'),
                'model': (None, 'whisper-1'),
                'language': (None, 'en')
            }

            response = requests.post(
                'https://api.openai.com/v1/audio/transcriptions',
                headers=headers,
                files=files
            )

        # Clean up temp file
        os.unlink(tmp_path)

        if response.status_code == 200:
            result = response.json()
            return result.get('text', '')
        else:
            print(f"OpenAI Whisper API error: {response.status_code}")
            return None

    except Exception as e:
        print(f"Error in OpenAI transcription: {e}")
        return None

def transcribe_audio_groq(audio_data, sample_rate):
    """Transcribe audio using Groq's Whisper API - very fast!"""

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        # Convert numpy array to WAV bytes
        audio_np = np.array(audio_data).astype(np.float32)

        # Normalize audio to 16-bit PCM
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_np * 32767).astype(np.int16)

        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            wavfile.write(tmp_file.name, sample_rate, audio_int16)
            tmp_path = tmp_file.name

        # Send to Groq Whisper API
        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        with open(tmp_path, 'rb') as audio_file:
            files = {
                'file': ('audio.wav', audio_file, 'audio/wav'),
                'model': (None, 'whisper-large-v3'),
                'language': (None, 'en')
            }

            response = requests.post(
                'https://api.groq.com/openai/v1/audio/transcriptions',
                headers=headers,
                files=files
            )

        # Clean up temp file
        os.unlink(tmp_path)

        if response.status_code == 200:
            result = response.json()
            return result.get('text', '')
        else:
            print(f"Groq Whisper API error: {response.status_code}")
            return None

    except Exception as e:
        print(f"Error in Groq transcription: {e}")
        return None