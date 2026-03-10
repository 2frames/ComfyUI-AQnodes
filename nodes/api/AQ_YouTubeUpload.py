import os
import requests


_INSTRUCTION = """AQ_YouTubeUpload — How to get credentials (one-time setup)
=================================================================

YouTube uploads require OAuth 2.0, NOT a plain API key.
You need three values: Client ID, Client Secret, Refresh Token.

STEP 1 — Create a Google Cloud project & enable YouTube API
-----------------------------------------------------------
1. Go to https://console.cloud.google.com/
2. Create a new project (top bar → "New Project").
3. Navigate to: APIs & Services → Library
4. Search "YouTube Data API v3" → click it → Enable.

STEP 2 — Create OAuth 2.0 credentials
--------------------------------------
1. Navigate to: APIs & Services → Credentials
2. Click "Create Credentials" → "OAuth client ID"
3. If prompted, configure the OAuth consent screen first:
   - User type: External
   - Fill in app name, your email → Save
   - Scopes: add "YouTube Data API v3 → .../auth/youtube.upload"
   - Add yourself as a Test User (your Google account email)
4. Back in Credentials → Create Credentials → OAuth client ID:
   - Application type: "Desktop app"
   - Give it a name → Create
5. Copy "Client ID" and "Client Secret" from the dialog.

STEP 3 — Generate a Refresh Token (one-time)
--------------------------------------------
1. Visit https://developers.google.com/oauthplayground/
2. Click the gear icon (⚙️) in the top-right:
   - Check "Use your own OAuth credentials"
   - Enter your Client ID and Client Secret → Close
3. In "Step 1 — Select & authorize APIs":
   - Paste or find: https://www.googleapis.com/auth/youtube.upload
   - Click "Authorize APIs"
   - Sign in with the Google account that owns the YouTube channel
4. In "Step 2 — Exchange authorization code for tokens":
   - Click "Exchange authorization code for tokens"
   - Copy the "Refresh token" value

STEP 4 — Fill in the node
--------------------------
- client_id     → your OAuth Client ID
- client_secret → your OAuth Client Secret
- refresh_token → the refresh token from Step 3

The refresh token does not expire unless you revoke it.
You only need to do this setup once per YouTube channel.

PRIVACY STATUS
--------------
- private   → only you can see it
- unlisted  → anyone with the link can see it
- public    → visible to everyone

DESCRIPTION AUTO-BUILD
-----------------------
If you leave 'description' empty, the node auto-composes it
from tagline, tags, and keyscale inputs.
"""


class AQ_YouTubeUpload:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path":     ("STRING",  {"default": ""}),
                "client_id":      ("STRING",  {"default": ""}),
                "client_secret":  ("STRING",  {"default": ""}),
                "refresh_token":  ("STRING",  {"default": ""}),
                "title":          ("STRING",  {"default": "My Song"}),
                "privacy_status": (["private", "unlisted", "public"], {"default": "private"}),
            },
            "optional": {
                "description": ("STRING",  {"multiline": True,  "default": ""}),
                "tagline":     ("STRING",  {"multiline": False, "default": ""}),
                "headline":    ("STRING",  {"multiline": False, "default": ""}),
                "tags":        ("STRING",  {"multiline": False, "default": ""}),
                "keyscale":    ("STRING",  {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "video_id", "instruction")
    OUTPUT_NODE = True
    FUNCTION = "upload"
    CATEGORY = "Aquasite/Media"

    def upload(self, video_path, client_id, client_secret, refresh_token, title,
               privacy_status="private", description="", tagline="", headline="",
               tags="", keyscale=""):

        if not all([video_path, client_id, client_secret, refresh_token]):
            return ("", "", _INSTRUCTION)

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Auto-compose description if not provided
        if not description.strip():
            parts = []
            if tagline:
                parts.append(tagline)
            if tags:
                parts.append(f"Style: {tags}")
            if keyscale:
                parts.append(f"Key: {keyscale}")
            if headline:
                parts.append(f"\n{headline}")
            description = "\n".join(parts)

        access_token = self._get_access_token(client_id, client_secret, refresh_token)

        tags_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

        video_id = self._resumable_upload(
            access_token=access_token,
            video_path=video_path,
            title=title,
            description=description,
            tags=tags_list,
            privacy_status=privacy_status,
            category_id="10",       # 10 = Music
        )

        video_url = f"https://www.youtube.com/watch?v={video_id}"
        print(f"[AQ_YouTubeUpload] Uploaded: {video_url}")
        return (video_url, video_id, _INSTRUCTION)

    # ------------------------------------------------------------------

    def _get_access_token(self, client_id, client_secret, refresh_token):
        resp = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id":     client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type":    "refresh_token",
            },
            timeout=30,
        )
        if not resp.ok:
            raise RuntimeError(f"Failed to get access token: {resp.status_code} {resp.text}")
        return resp.json()["access_token"]

    def _resumable_upload(self, access_token, video_path, title, description,
                          tags, privacy_status, category_id):
        file_size = os.path.getsize(video_path)

        metadata = {
            "snippet": {
                "title":       title,
                "description": description,
                "tags":        tags,
                "categoryId":  category_id,
            },
            "status": {
                "privacyStatus": privacy_status,
            },
        }

        # Initiate upload session
        init = requests.post(
            "https://www.googleapis.com/upload/youtube/v3/videos",
            params={"uploadType": "resumable", "part": "snippet,status"},
            headers={
                "Authorization":          f"Bearer {access_token}",
                "Content-Type":           "application/json; charset=UTF-8",
                "X-Upload-Content-Type":  "video/mp4",
                "X-Upload-Content-Length": str(file_size),
            },
            json=metadata,
            timeout=60,
        )
        if not init.ok:
            raise RuntimeError(f"Upload init failed: {init.status_code} {init.text}")

        upload_url = init.headers["Location"]

        # Upload in 10 MB chunks
        chunk_size = 10 * 1024 * 1024
        with open(video_path, "rb") as fh:
            start = 0
            while start < file_size:
                chunk = fh.read(chunk_size)
                end = start + len(chunk) - 1

                put = requests.put(
                    upload_url,
                    headers={
                        "Content-Range": f"bytes {start}-{end}/{file_size}",
                        "Content-Type":  "video/mp4",
                    },
                    data=chunk,
                    timeout=300,
                )

                if put.status_code in (200, 201):
                    return put.json()["id"]
                elif put.status_code == 308:        # Resume Incomplete
                    range_header = put.headers.get("Range", "")
                    if range_header:
                        start = int(range_header.split("-")[1]) + 1
                    else:
                        start = end + 1
                else:
                    raise RuntimeError(
                        f"Upload chunk failed: {put.status_code} {put.text}"
                    )

        raise RuntimeError("Upload ended without receiving a video ID")
