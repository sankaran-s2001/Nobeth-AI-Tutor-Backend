import os
import re
import logging
from django.http import HttpResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from groq import Groq

logger = logging.getLogger(__name__)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def transcribe_view(request):
    """
    POST /api/transcribe or /api/audio/transcribe
    Transcribes uploaded audio files using Groq Whisper model.
    """
    audio_file = request.FILES.get('file')
    if not audio_file:
        return Response(
            {"error": "No audio file provided in request. Please upload a file with the key 'file'."},
            status=status.HTTP_400_BAD_REQUEST
        )
        
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("[AUDIO] GROQ_API_KEY is not set.")
        return Response(
            {"error": "Speech service configuration error on server."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        
    try:
        client = Groq(api_key=api_key)
        file_bytes = audio_file.read()
        
        # Try models in order of preference
        whisper_models = ["whisper-large-v3-turbo", "whisper-large-v3"]
        transcription = None
        last_err = None
        
        for model in whisper_models:
            try:
                transcription = client.audio.transcriptions.create(
                    file=(audio_file.name or "audio.webm", file_bytes),
                    model=model
                )
                break  # Success!
            except Exception as e:
                logger.warning(f"[AUDIO] Transcription failed with model {model}: {e}")
                last_err = e
                continue
                
        if not transcription:
            raise last_err
            
        return Response({"text": transcription.text})
    except Exception as e:
        logger.exception(f"[AUDIO] Transcription failed: {e}")
        err_msg = str(e)
        if "blocked at the organization level" in err_msg:
            err_msg += " (Please enable this model in your Groq Org settings at https://console.groq.com/settings/limits)"
        return Response(
            {"error": f"Transcription failed: {err_msg}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def speak_view(request):
    """
    GET /api/speak or /api/audio/speak
    Generates WAV audio bytes using Groq Orpheus TTS model.
    """
    text = request.GET.get('text', '').strip()
    if not text:
        return Response(
            {"error": "No text provided. Please pass text in the 'text' query parameter."},
            status=status.HTTP_400_BAD_REQUEST
        )
        
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("[AUDIO] GROQ_API_KEY is not set.")
        return Response(
            {"error": "Speech service configuration error on server."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        
    try:
        # Strip <think>...</think> reasoning tags
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        
        # If text becomes empty after stripping reasoning, return warning or empty response
        if not cleaned_text:
            return Response(
                {"error": "Provided text contained only internal reasoning tags."},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        client = Groq(api_key=api_key)
        response = client.audio.speech.create(
            model="canopylabs/orpheus-v1-english",
            voice="daniel",
            input=cleaned_text,
            response_format="wav"
        )
        
        django_res = HttpResponse(response.read(), content_type="audio/wav")
        django_res['Content-Disposition'] = 'attachment; filename="speech.wav"'
        return django_res
    except Exception as e:
        logger.exception(f"[AUDIO] Text-to-Speech generation failed: {e}")
        err_msg = str(e)
        if "requires terms acceptance" in err_msg:
            err_msg += " (Please accept the terms for canopylabs/orpheus-v1-english model in your Groq account at https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english)"
        return Response(
            {"error": f"TTS generation failed: {err_msg}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
