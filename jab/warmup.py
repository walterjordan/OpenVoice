"""Runs at image build so every model download lands in the image, not on a cold start."""
import os

import nltk

nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('cmudict', quiet=True)

os.environ.setdefault('OPENVOICE_TOKEN', 'warmup')

from fastapi.testclient import TestClient

from jab.server import ROOT, app

client = TestClient(app)
auth = {'Authorization': 'Bearer warmup'}
with open(os.path.join(ROOT, 'resources', 'example_reference.mp3'), 'rb') as f:
    r = client.post('/v1/embed', content=f.read(), headers=auth)
print(r.status_code, r.text[:500]) if r.status_code >= 400 else None
r.raise_for_status()
emb = r.json()
print('embed ok', emb['shape'], emb['speechSeconds'])
r = client.post('/v1/speak', json={'text': 'Warm up complete.', 'embedding': emb['embedding']}, headers=auth)
print(r.status_code, r.text[:500]) if r.status_code >= 400 else None
r.raise_for_status()
print('speak ok', len(r.content), 'bytes')
