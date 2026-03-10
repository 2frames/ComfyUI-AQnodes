import os
import requests

_KEYSCALES = [
    f"{root} {quality}"
    for quality in ["major", "minor"]
    for root in ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"]
]


_INSTRUCTION = """AQ_YouTubeUpload — How to get credentials (one-time setup)
=================================================================

YouTube uploads require OAuth 2.0, NOT a plain API key.
You need three values: Client ID, Client Secret, Refresh Token.

STEP 1 — Create a Google Cloud project & enable YouTube API
-----------------------------------------------------------
1. Go to https://console.cloud.google.com/
2. Create a new project (top bar -> "New Project").
3. Navigate to: APIs & Services -> Library
4. Search "YouTube Data API v3" -> click it -> Enable.

STEP 2 — Configure OAuth consent screen
----------------------------------------
1. Navigate to: APIs & Services -> OAuth consent screen
2. User type: External -> Create
3. Fill in App name and your email address -> Save and Continue
4. On the Scopes screen -> Add or Remove Scopes ->
   manually type: https://www.googleapis.com/auth/youtube.upload
   -> Add to table -> Update -> Save and Continue
5. On the Test users screen -> Add Users ->
   add the Google account email that owns your YouTube channel
   -> Save and Continue

STEP 3 — Create OAuth 2.0 credentials (WEB APPLICATION type)
--------------------------------------------------------------
!! IMPORTANT: use "Web application" type, NOT "Desktop app" !!
The OAuth Playground requires a web redirect URI. Desktop app
clients do not support it and will cause "unauthorized_client".

1. Navigate to: APIs & Services -> Credentials
2. Click "Create Credentials" -> "OAuth client ID"
3. Application type: "Web application"
4. Under "Authorized redirect URIs" -> Add URI:
       https://developers.google.com/oauthplayground
   (exactly as written, no trailing slash)
5. Click Create -> copy the Client ID and Client Secret.

STEP 4 — Generate a Refresh Token via OAuth Playground
-------------------------------------------------------
!! IMPORTANT: configure your credentials in the Playground
   BEFORE clicking "Authorize APIs" — order matters !!

1. Open https://developers.google.com/oauthplayground/
2. Click the gear icon top-right:
   - Check "Use your own OAuth credentials"
   - Paste your Client ID and Client Secret
   - Close the settings panel
3. In "Step 1 — Select & authorize APIs" (left panel):
   - In the input box at the bottom type:
         https://www.googleapis.com/auth/youtube.upload
   - Click "Authorize APIs"
   - Sign in with the Google account added as Test User in Step 2
   - Click "Allow"
4. You are now in "Step 2 — Exchange authorization code for tokens":
   - Click "Exchange authorization code for tokens"
   - Copy the "Refresh token" value (starts with "1//" or "1/")

STEP 5 — Fill in the node
--------------------------
  client_id     -> Client ID from Step 3
  client_secret -> Client Secret from Step 3
  refresh_token -> Refresh token from Step 4

Refresh tokens do not expire unless revoked.
This setup is needed only once per YouTube channel.

TROUBLESHOOTING
---------------
"unauthorized_client":
  Most likely cause A: you created a "Desktop app" client instead
  of "Web application". Re-create the OAuth client as Web app and
  add the Playground redirect URI (Step 3 above), then redo Step 4.

  Most likely cause B: you forgot to set your own credentials in
  the Playground gear icon BEFORE authorizing. The token you got
  belongs to Google's own Playground client, not yours.
  Re-do Step 4 — set credentials in gear icon first, then Authorize.

"access_denied" or 403 Forbidden during upload:
  The uploading Google account is not listed as a Test User on the
  OAuth consent screen. Add it (Step 2, item 5) and redo Step 4.

"invalid_grant":
  The refresh token has been revoked or the authorization was
  removed. Re-generate starting from Step 4.

PRIVACY STATUS
--------------
  private   -> only you can see it
  unlisted  -> anyone with the link can see it
  public    -> visible to everyone

DESCRIPTION AUTO-BUILD
-----------------------
If you leave description empty, the node auto-composes it from
tagline, tags, and keyscale inputs.
"""


class AQ_YouTubeUpload:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path":     ("STRING", {"default": ""}),
                "client_id":      ("STRING", {"default": ""}),
                "client_secret":  ("STRING", {"default": ""}),
                "refresh_token":  ("STRING", {"default": ""}),
                "title":          ("STRING", {"default": "My Song"}),
                "privacy_status": (["private", "unlisted", "public"], {"default": "private"}),
            },
            "optional": {
                "description": ("STRING", {"multiline": True,  "default": ""}),
                "tagline":     ("STRING", {"multiline": False, "default": ""}),
                "headline":    ("STRING", {"multiline": False, "default": ""}),
                "tags":        ("STRING", {"multiline": False, "default": ""}),
                "keyscale":    (_KEYSCALES, {"default": "C major"}),
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
            category_id="10",   # 10 = Music
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
            body = resp.text
            hint = ""
            if "unauthorized_client" in body:
                hint = (
                    "\nFix: re-create the OAuth client as 'Web application' type "
                    "(not Desktop app) and add https://developers.google.com/oauthplayground "
                    "as an authorized redirect URI, then regenerate the refresh token. "
                    "Also make sure you set your own credentials in the Playground gear "
                    "icon BEFORE clicking Authorize APIs."
                )
            elif "invalid_grant" in body:
                hint = "\nFix: the refresh token was revoked. Re-generate it via OAuth Playground."
            raise RuntimeError(
                f"Failed to get access token ({resp.status_code}): {body}{hint}"
            )
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

        init = requests.post(
            "https://www.googleapis.com/upload/youtube/v3/videos",
            params={"uploadType": "resumable", "part": "snippet,status"},
            headers={
                "Authorization":           f"Bearer {access_token}",
                "Content-Type":            "application/json; charset=UTF-8",
                "X-Upload-Content-Type":   "video/mp4",
                "X-Upload-Content-Length": str(file_size),
            },
            json=metadata,
            timeout=60,
        )
        if not init.ok:
            raise RuntimeError(f"Upload init failed ({init.status_code}): {init.text}")

        upload_url = init.headers["Location"]

        chunk_size = 10 * 1024 * 1024  # 10 MB
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
                elif put.status_code == 308:    # Resume Incomplete
                    range_header = put.headers.get("Range", "")
                    start = int(range_header.split("-")[1]) + 1 if range_header else end + 1
                else:
                    raise RuntimeError(
                        f"Upload chunk failed ({put.status_code}): {put.text}"
                    )

        raise RuntimeError("Upload ended without receiving a video ID")
