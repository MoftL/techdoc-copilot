# Deployment

The app is designed for [Streamlit Community Cloud](https://streamlit.io/cloud), which deploys directly from GitHub.

## Before you deploy

1. **Set a spending limit** in your [OpenAI billing settings](https://platform.openai.com/settings/organization/limits). The app's own limits reset on restart, so this is your real safety net.
2. **Confirm your key isn't in the repository.** `.env` is listed in `.gitignore`, and should never be committed.

## Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. Choose **Create app**, then select this repository and the `main` branch.
3. Set the main file path to `src/app.py`.
4. Under **Advanced settings**, choose Python 3.13 and add your key under **Secrets**:

   ```toml
   OPENAI_API_KEY = "your-key-here"
   ```

5. Deploy. The first build and the first question take longer, because dependencies and the reranker model are downloaded.

Streamlit exposes top-level secrets as environment variables, so the app needs no code changes.

## Settings

Set any of these as secrets or environment variables.

| Setting | Default | What it controls |
|---|---|---|
| `MAX_QUESTIONS_PER_SESSION` | 20 | Questions per visitor session |
| `MAX_QUESTIONS_PER_DAY` | 300 | Questions per day, across all visitors |
| `MAX_UPLOADS_PER_SESSION` | 3 | Uploads per visitor session |
| `MAX_UPLOADS_PER_DAY` | 30 | Uploads per day, across all visitors |
| `MAX_UPLOAD_TOKENS` | 500000 | Largest upload accepted |
| `CONTEXT_TOKEN_LIMIT` | 40000 | Uploads above this skip the AI context step |
| `MAX_WORKSPACES` | 3 | Uploads kept in memory at once |

## If the app runs out of memory

The app peaked at about 670 MB in testing, but several large uploads at once could use more. Try these, in order:

1. Set `MAX_WORKSPACES` to `1` or `2`.
2. Lower `MAX_UPLOAD_TOKENS`.
3. Move to a host with more memory.

## Hosting the demo page

The interactive page in `docs/` is static and needs no server. In the repository's **Settings > Pages**, deploy from the `main` branch and the `/docs` folder.

To refresh its data after changing a pipeline, run `python scripts/generate_demo_data.py`.
