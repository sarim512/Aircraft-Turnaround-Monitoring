# """Resolve a YouTube (live) page URL into a directly readable stream URL via yt-dlp."""
# from __future__ import annotations

# import logging
# from typing import Any, Mapping

# logger = logging.getLogger(__name__)


# class StreamResolutionError(RuntimeError):
#     """Raised when yt-dlp cannot produce a playable stream URL."""


# def pick_stream_url(info: Mapping[str, Any]) -> str:
#     """Choose the playable URL from a yt-dlp ``extract_info`` result.

#     Args:
#         info: Dictionary returned by ``YoutubeDL.extract_info(download=False)``.
#     Returns:
#         A direct media/manifest URL.
#     Raises:
#         StreamResolutionError: If no usable URL exists in ``info``.
#     Logic:
#         1. Prefer top-level ``info["url"]`` (single selected format).
#         2. Else the first ``requested_formats`` entry that has a URL.
#         3. Else the last entry of ``formats`` that has a URL (yt-dlp sorts worst -> best).
#     """
#     url = info.get("url")
#     if url:
#         return str(url)
#     for fmt in info.get("requested_formats") or []:
#         if fmt.get("url"):
#             return str(fmt["url"])
#     for fmt in reversed(info.get("formats") or []):
#         if fmt.get("url"):
#             return str(fmt["url"])
#     raise StreamResolutionError("yt-dlp returned no playable URL")


# def resolve_youtube_stream(page_url: str, max_height: int = 720) -> str:
#     """Resolve a YouTube page URL to a stream URL.

#     Args:
#         page_url: ``https://www.youtube.com/watch?v=...`` (live or VOD) URL.
#         max_height: Preferred maximum video height in pixels.
#     Returns:
#         Direct stream URL (typically an HLS ``.m3u8`` for live streams). It expires, so callers
#         must re-resolve on reconnect.
#     Raises:
#         StreamResolutionError: yt-dlp missing, extraction failed, or no URL found.
#     Logic:
#         1. Import yt_dlp lazily.
#         2. Format selector prefers muxed HLS up to ``max_height``, then any format up to it, then best.
#         3. ``extract_info(download=False)`` then ``pick_stream_url``.
#     Side effects:
#         Network access to YouTube through yt-dlp.
#     """
#     try:
#         import yt_dlp  # type: ignore
#     except ImportError as exc:
#         raise StreamResolutionError("yt-dlp is not installed (pip install yt-dlp)") from exc
#     options = {
#         "quiet": True,
#         "no_warnings": True,
#         "noplaylist": True,
#         "format": f"best[height<={max_height}][protocol^=m3u8]/best[height<={max_height}]/best",
#     }
#     try:
#         with yt_dlp.YoutubeDL(options) as ydl:
#             info = ydl.extract_info(page_url, download=False)
#     except Exception as exc:  # yt-dlp raises many distinct exception types
#         raise StreamResolutionError(f"yt-dlp failed for {page_url}: {exc}") from exc
#     url = pick_stream_url(info)
#     logger.info("Resolved %s -> stream URL (%d chars)", page_url, len(url))
#     return url
























"""Resolve a YouTube (live) page URL into a directly readable stream URL via yt-dlp."""
from __future__ import annotations

import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)


class StreamResolutionError(RuntimeError):
    """Raised when yt-dlp cannot produce a playable stream URL."""


def pick_stream_url(info: Mapping[str, Any]) -> str:
    """Choose the playable URL from a yt-dlp ``extract_info`` result.

    Args:
        info: Dictionary returned by ``YoutubeDL.extract_info(download=False)``.
    Returns:
        A direct media/manifest URL.
    Raises:
        StreamResolutionError: If no usable URL exists in ``info``.
    Logic:
        1. Prefer top-level ``info["url"]`` (single selected format).
        2. Else the first ``requested_formats`` entry that has a URL.
        3. Else the last entry of ``formats`` that has a URL (yt-dlp sorts worst -> best).
    """
    url = info.get("url")
    if url:
        return str(url)
    for fmt in info.get("requested_formats") or []:
        if fmt.get("url"):
            return str(fmt["url"])
    for fmt in reversed(info.get("formats") or []):
        if fmt.get("url"):
            return str(fmt["url"])
    raise StreamResolutionError("yt-dlp returned no playable URL")


def resolve_youtube_stream(page_url: str, max_height: int = 720) -> str:
    """Resolve a YouTube page URL to a stream URL.

    Args:
        page_url: ``https://www.youtube.com/watch?v=...`` (live or VOD) URL.
        max_height: Preferred maximum video height in pixels.
    Returns:
        Direct stream URL (typically an HLS ``.m3u8`` for live streams or direct MP4 stream).
    Raises:
        StreamResolutionError: yt-dlp missing, extraction failed, or no URL found.
    """
    try:
        import yt_dlp  # type: ignore
    except ImportError as exc:
        raise StreamResolutionError("yt-dlp is not installed (pip install yt-dlp)") from exc

    # Updated format selector:
    # 1. Muxed HLS stream <= max_height (for live streams)
    # 2. Standalone video stream <= max_height (for regular videos / OpenCV)
    # 3. Any best video stream available
    format_str = (
        f"best[height<={max_height}][protocol^=m3u8]/"
        f"bestvideo[height<={max_height}]/"
        f"best[height<={max_height}]/"
        f"bv*/best"
    )

    options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": format_str,
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(page_url, download=False)
    except Exception as exc:  # yt-dlp raises many distinct exception types
        raise StreamResolutionError(f"yt-dlp failed for {page_url}: {exc}") from exc

    url = pick_stream_url(info)
    logger.info("Resolved %s -> stream URL (%d chars)", page_url, len(url))
    return url