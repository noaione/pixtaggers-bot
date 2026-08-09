# pixtaggers-bot

A simple webhook-based tagging bot for szurubooru compatible boorus.

This bot can:
- Automatically tag posts with CamieTagger V2 or CL Tagger V2 based on the image content.
  - Also support videos by capturing multiple frames and generating tags based on them.
- Manually tag posts by sending a POST request to the `/tag` endpoint with the post ID.
- Generate proper thumbnails for video, avoid any single color frames.
- Generate proper alpha thumbnails for images with alpha channel.

**Why make another one?**

Idk, haven't checked the other tools available and I already made it myself in script format, this version is more complete with proper API endpoint.

## Configuration

The bot is configured using a config.json file. You can use the provided config.example.json as a template.

## Usage

1. Install uv
2. Run `uv sync --locked` to install dependencies
3. Run `uv run uvicorn main:app --host <IP_ADDRESS> --port 42069` to start the bot
4. Configure your booru to send webhooks to `http://<bot-ip>:42069/webhook?t=<key-from-config>` for new posts and `http://<bot-ip>:42069/tag` for manual tagging.
5. The bot will automatically tag new posts based on the image content and will also respond to manual tagging requests.

**Note**: You would always need to provide the query parameter `t` with the value of the `key` from the config for authentication when sending requests to the bot.

This is used to verify that the request is coming from an authorized source and to prevent unauthorized access to the bot's functionality.

Tag list startup cache
----------------------

The complete Szurubooru tag list is stored at `.cache/szurubooru-tags.json` after first fetch. Later starts load it first, fetch tags sorted by creation time, stop after finding a cached tag, merge results, and save the cache again. Configure `szuru.tag_cache_path` in `config.json` when needed.

HTTP backend
------------

Default backend is `httpx`. Set `szuru.http_impersonate` to a supported target such as `chrome`, `chrome131`, or `safari` to automatically use `curl_cffi`. Leave it `null` or empty to keep `httpx`. `curl_cffi` handles browser-style TLS and HTTP/2 fingerprints.

## Where to download models?

Set the `model` value in `config.json` to select a model:

- `camie-tagger-v2`
- `cl-tagger-v2`

### CamieTagger V2

Get `camie-tagger-v2.onnx` and `camie-tagger-v2-metadata.json` from [CamieTagger V2 on Hugging Face](https://huggingface.co/Camais03/camie-tagger-v2/tree/main).

Download and place them in the `./pixtaggers/models/camie-tagger-v2` directory.
Or use the Hugging Face CLI:

```bash
hf download hf://Camais03/camie-tagger-v2/camie-tagger-v2.onnx --local-dir ./pixtaggers/models/camie-tagger-v2
hf download hf://Camais03/camie-tagger-v2/camie-tagger-v2-metadata.json --local-dir ./pixtaggers/models/camie-tagger-v2
```

### CL Tagger V2

Download the following files from [CL Tagger V2 on Hugging Face](https://huggingface.co/cella110n/cl_tagger_v2) into `./pixtaggers/models/cl-tagger-v2`:

- `model.onnx`
- `model.onnx.data`
- `model_metadata.json`
- `model_vocabulary.json`
- `model_tag_metrics.npz`
- `model_ood_ref.npz`

The repository requires accepting its access conditions before downloading. Using the Hugging Face CLI:

```bash
hf download hf://cella110n/cl_tagger_v2/v2_01a/model.onnx --local-dir ./pixtaggers/models/cl-tagger-v2
hf download hf://cella110n/cl_tagger_v2/v2_01a/model.onnx.data --local-dir ./pixtaggers/models/cl-tagger-v2
hf download hf://cella110n/cl_tagger_v2/v2_01a/model_metadata.json --local-dir ./pixtaggers/models/cl-tagger-v2
hf download hf://cella110n/cl_tagger_v2/v2_01a/model_vocabulary.json --local-dir ./pixtaggers/models/cl-tagger-v2
hf download hf://cella110n/cl_tagger_v2/v2_01a/model_ood_ref.npz --local-dir ./pixtaggers/models/cl-tagger-v2
hf download hf://cella110n/cl_tagger_v2/v2_01a/model_tag_metrics.npz --local-dir ./pixtaggers/models/cl-tagger-v2
```
