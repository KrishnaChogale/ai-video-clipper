# Jo Koy: Blue in the Face Campaign Clipper

Separate local clipping app for this campaign. It has its own `input`, `output`, `temp`, `logs`, and `database` folders, so running it does not mix campaign clips with the GTA project.

## Run it

From PowerShell:

```powershell
cd C:\ai-video-clipper\campaigns\jo-koy-blue-in-the-face-clipper
..\..\venv\Scripts\python.exe -m streamlit run app.py
```

Or run `run_campaign.ps1` from this folder. The runner reuses the parent project's installed Python packages; the campaign files and generated data remain separate. Install dependencies from this folder's `requirements.txt` if you want a fully independent Python environment.

## Workflow

1. Add the source performance video in the app.
2. Choose the number of clips, duration mode, comedy style, and caption settings.
3. Select **Find, Cut & Caption**. The app analyzes the video, selects non-overlapping moments, cuts the clips, and generates timed captions from the speech transcript.
4. Find exports and metadata in this folder's `output` directory.

The app only exports candidates scoring **70/100 or higher** on its local evidence heuristic. This is a selection threshold, not a measured probability of going viral; if too few moments qualify, it exports fewer clips instead of lowering the bar. Around one fifth of selected clips get a transcript-based `POV:` opening hook over the blurred 9:16 background treatment; the remaining clips keep the selected framing.

The comedy cues use transcript text, speech timing, audio energy, and visual motion. Explicit laughter words in the transcript are treated as a cue; audio peaks alone are only a possible-reaction signal. This is a local heuristic pipeline, not a model that understands every joke. Use the actual campaign footage to assess which moments it selects.

The local Ollama model is optional. Faster-Whisper, FFmpeg, YOLO, and the Python packages listed in `requirements.txt` are used by the pipeline. The bundled `yolo11n.pt` is the lightweight object detector copied from the base project.

## Separation

- The original GTA clipper remains in the project root.
- This campaign uses the same modular architecture, but campaign-specific config, comedy metadata, and laughter/reaction cues.
- Campaign inputs, exports, and SQLite history are stored only under this folder.
- No videos, database, output clips, or cookie files were copied into this campaign folder.
