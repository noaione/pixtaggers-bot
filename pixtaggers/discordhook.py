import httpx


class DiscordHook:
    def __init__(self, webhook_url: str | None, *, host_urL: str):
        self.webhook_url = webhook_url
        self.host_url = host_urL

    async def report_error(self, post_id: int, error_message: str):
        if not self.webhook_url:
            return

        # Make into embed format
        embed = {
            "title": f"Error processing post #{post_id}",
            "description": error_message,
            "color": 0xFF0000,  # Red color
            # Set the title to be clickable and link to the post on the booru
            "url": f"{self.host_url}/post/{post_id}",
            "author": {
                "name": "Pixtaggers",
                "url": self.host_url,
            }
        }

        async with httpx.AsyncClient() as client:
            try:
                await client.post(self.webhook_url, json={"embeds": [embed]})
            except Exception as e:
                print(f"Failed to send error report to Discord: {e}")
