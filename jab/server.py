"""JAB voice-clone service on OpenVoice V2.

Stateless by design: /v1/embed turns a consented voice sample into a tone-color
embedding (256 floats) and hands it back; the caller stores it next to the
account that owns the voice. /v1/speak takes that embedding with the text, so
this service never holds anyone's voice and needs no bucket or database.
"""
import base64
import hmac
import io
import os
import subprocess
import tempfile
import threading

import librosa
import numpy as np
import soundfile
import torch
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from openvoice.api import ToneColorConverter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.environ.get('OPENVOICE_CKPT', os.path.join(ROOT, 'checkpoints_v2'))
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
TOKEN = os.environ.get('OPENVOICE_TOKEN', '').strip()
WATERMARK = os.environ.get('OPENVOICE_WATERMARK', '@JAB')
MAX_CHARS = 3000
MAX_SAMPLE_BYTES = 25 * 1024 * 1024
MIN_SPEECH_SECONDS = 6.0

LANGUAGES = {
    'EN': ['EN-US', 'EN-BR', 'EN_INDIA', 'EN-AU', 'EN-Default'],
    'ES': ['ES'], 'FR': ['FR'], 'ZH': ['ZH'], 'JP': ['JP'], 'KR': ['KR'],
}

converter = ToneColorConverter(f'{CKPT}/converter/config.json', device=DEVICE)
converter.load_ckpt(f'{CKPT}/converter/checkpoint.pth')

# MeloTTS and the converter are not safe to drive from two requests at once on
# one model instance; Cloud Run concurrency is 1, the lock covers local runs.
_lock = threading.Lock()
_tts = {}
_source_se = {}

app = FastAPI(title='jab-openvoice')


def _melo(language: str):
    if language not in _tts:
        from melo.api import TTS
        _tts[language] = TTS(language=language, device=DEVICE)
    return _tts[language]


def _base_se(speaker_key: str) -> torch.Tensor:
    key = speaker_key.lower().replace('_', '-')
    if key not in _source_se:
        _source_se[key] = torch.load(f'{CKPT}/base_speakers/ses/{key}.pth', map_location=DEVICE)
    return _source_se[key]


def _auth(authorization: str | None):
    if not TOKEN:
        raise HTTPException(503, 'OPENVOICE_TOKEN is not configured')
    given = (authorization or '').removeprefix('Bearer ').strip()
    if not hmac.compare_digest(given, TOKEN):
        raise HTTPException(401, 'unauthorized')


def _encode(audio: np.ndarray, sr: int, fmt: str) -> tuple[bytes, str]:
    wav = io.BytesIO()
    soundfile.write(wav, audio, sr, format='WAV')
    if fmt == 'wav':
        return wav.getvalue(), 'audio/wav'
    args = {
        'mp3': (['-f', 'mp3', '-b:a', '128k'], 'audio/mpeg'),
        # Twilio media streams and <Play> both take 8 kHz mu-law.
        'mulaw': (['-ar', '8000', '-ac', '1', '-f', 'mulaw'], 'audio/basic'),
    }
    if fmt not in args:
        raise HTTPException(400, f'unsupported format {fmt}')
    flags, mime = args[fmt]
    out = subprocess.run(['ffmpeg', '-v', 'error', '-i', 'pipe:0', *flags, 'pipe:1'],
                         input=wav.getvalue(), capture_output=True, check=True)
    return out.stdout, mime


@app.get('/healthz')
def healthz():
    return {'ok': True, 'device': DEVICE, 'loaded': sorted(_tts)}


@app.post('/v1/embed')
async def embed(request: Request, authorization: str | None = Header(None)):
    """Body: raw audio bytes (any format ffmpeg reads). Returns the embedding."""
    _auth(authorization)
    body = await request.body()
    if not body:
        raise HTTPException(400, 'send the voice sample as the request body')
    if len(body) > MAX_SAMPLE_BYTES:
        raise HTTPException(413, 'sample is larger than 25 MB')
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, 'sample.wav')
        try:
            subprocess.run(['ffmpeg', '-v', 'error', '-i', 'pipe:0', '-ac', '1', '-ar', '22050', src],
                           input=body, capture_output=True, check=True)
        except subprocess.CalledProcessError:
            raise HTTPException(422, 'could not read that audio file')
        seconds = soundfile.info(src).duration
        sr = converter.hps.data.sampling_rate
        y, _ = librosa.load(src, sr=sr)
        voiced = [y[s:e] for s, e in librosa.effects.split(y, top_db=35)]
        active = np.concatenate(voiced) if voiced else np.zeros(0, dtype=np.float32)
        speech = len(active) / sr
        if speech < MIN_SPEECH_SECONDS:
            raise HTTPException(422, f'only {speech:.0f}s of speech found; record 30 to 60 seconds')
        # Same shape as upstream get_se: ~10 s segments, embeddings averaged.
        paths = []
        for i, chunk in enumerate(np.array_split(active, max(1, round(speech / 10)))):
            path = os.path.join(tmp, f'seg{i}.wav')
            soundfile.write(path, chunk, sr)
            paths.append(path)
        with _lock:
            se = converter.extract_se(paths)
    vec = se.detach().cpu().float().numpy().astype('<f4')
    return {
        'embedding': base64.b64encode(vec.tobytes()).decode(),
        'shape': list(vec.shape),
        'model': 'openvoice-v2',
        'sampleSeconds': round(seconds, 1),
        'speechSeconds': round(speech, 1),
    }


@app.post('/v1/speak')
async def speak(request: Request, authorization: str | None = Header(None)):
    """JSON: {text, embedding, language?, speaker?, speed?, format?: mp3|wav|mulaw, tau?}."""
    _auth(authorization)
    body = await request.json()
    text = (body.get('text') or '').strip()
    if not text:
        raise HTTPException(400, 'text is required')
    if len(text) > MAX_CHARS:
        raise HTTPException(413, f'text is over {MAX_CHARS} characters')
    language = (body.get('language') or 'EN').upper()
    if language not in LANGUAGES:
        raise HTTPException(400, f'language must be one of {sorted(LANGUAGES)}')
    speaker = body.get('speaker') or LANGUAGES[language][0]
    speed = float(body.get('speed') or 1.0)
    tau = float(body.get('tau') or 0.3)
    fmt = body.get('format') or 'mp3'
    try:
        raw = np.frombuffer(base64.b64decode(body['embedding']), dtype='<f4')
        target = torch.from_numpy(raw.copy()).reshape(1, -1, 1).to(DEVICE)
    except Exception:
        raise HTTPException(400, 'embedding is missing or malformed')

    with tempfile.TemporaryDirectory() as tmp, _lock:
        model = _melo(language)
        ids = model.hps.data.spk2id
        if speaker not in ids:
            raise HTTPException(400, f'speaker must be one of {sorted(ids)}')
        base = os.path.join(tmp, 'base.wav')
        model.tts_to_file(text, ids[speaker], base, speed=speed, quiet=True)
        audio = converter.convert(audio_src_path=base, src_se=_base_se(speaker), tgt_se=target,
                                  tau=tau, message=WATERMARK)
    data, mime = _encode(audio, converter.hps.data.sampling_rate, fmt)
    return Response(content=data, media_type=mime)


@app.exception_handler(subprocess.CalledProcessError)
def ffmpeg_failed(_req, exc):
    return JSONResponse({'detail': f'audio encode failed: {exc.stderr[-300:]!r}'}, status_code=500)
