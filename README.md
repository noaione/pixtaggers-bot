# pixtaggers-bot

A simple webhook-based tagging bot for szurubooru compatible boorus.

This bot can:
- Automatically tag posts with CamieTagger V2 model based on the image content.
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

## Where to download models?

Get both the `camie-tagger-v2.onnx` and `camie-tagger-v2-metadata.json` file from here: https://huggingface.co/Camais03/camie-tagger-v2/tree/main

Download and place them in the `./pixtaggers/models/` directory.
