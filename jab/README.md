# jab-openvoice

OpenVoice V2 (MeloTTS base speakers + tone color converter) wrapped as a small HTTP
service for JAB client voice clones. Runs on Cloud Run as `jab-openvoice`
(project `gen-lang-client-0901205519`, `us-east1`).

The service is stateless. It never stores a recording or a voice. The caller (jab-ops)
keeps the person's recording in its private bucket and the 1 KB embedding on the
person's record, so deleting that record deletes the voice.

## API

All routes except `/healthz` need `Authorization: Bearer $OPENVOICE_TOKEN`
(secret `jab-openvoice-TOKEN`).

| Route | Body | Returns |
|---|---|---|
| `POST /v1/embed` | raw audio bytes, any format ffmpeg reads; 30 to 60 s of speech | `{ embedding, shape, model, sampleSeconds, speechSeconds }`; 422 when there is too little clear speech |
| `POST /v1/speak` | JSON `{ text, embedding, language?: EN, speaker?: EN-US, speed?: 1.0, format?: mp3 \| wav \| mulaw, tau?: 0.3 }` | audio bytes; `mulaw` is 8 kHz for Twilio |
| `GET /healthz` | | `{ ok, device, loaded }` |

Every output carries the OpenVoice audio watermark (`OPENVOICE_WATERMARK`, default `@JAB`).

## Build and deploy

```bash
gcloud builds submit . --tag us-east1-docker.pkg.dev/gen-lang-client-0901205519/cloud-run-source-deploy/jab-openvoice:<tag> --machine-type=e2-highcpu-8 --timeout=3600 --region=us-east1
gcloud run deploy jab-openvoice --image <that tag> --region=us-east1 --cpu=4 --memory=8Gi --concurrency=1 --max-instances=3 --min-instances=0 --timeout=300 --cpu-boost --allow-unauthenticated --set-secrets=OPENVOICE_TOKEN=jab-openvoice-TOKEN:latest
```

The image build runs `jab/warmup.py`, which downloads every model (MeloTTS, BERT,
silero VAD, the watermark model) so a cold start only loads from disk.
