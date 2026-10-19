import csv
import io
import json
from pathlib import Path
from typing import Any

from app.core.errors import InvalidInputException
from app.services.ingestion_schemas import AnalysisSubmission, IngestionResult, InputType
from app.agent.providers.youtube import YouTubeAdapter


class DataIngestionService:
    def __init__(self):
        self.yt_adapter = YouTubeAdapter()

    def process_submission(self, submission: AnalysisSubmission) -> IngestionResult:
        """Classify input target and ingest initial structured data."""
        target = submission.input_target.strip()

        # Check if target is a file in the uploads directory
        if target.startswith("upload://") or Path(target).exists():
            clean_path = target.replace("upload://", "")
            return self.process_file_upload(Path(clean_path))

        # Otherwise treat as YouTube URL or handle
        target_info = self.yt_adapter.parse_youtube_target(target)
        channel_data = self.yt_adapter.fetch_channel_data(target_info)

        custom_url = channel_data.get("custom_url")
        channel_id = channel_data.get("channel_id")
        if custom_url and custom_url.startswith("@"):
            channel_url = f"https://www.youtube.com/{custom_url}"
        elif channel_id:
            channel_url = f"https://www.youtube.com/channel/{channel_id}"
        else:
            channel_url = None

        analyzed_video = channel_data.get("analyzed_video")
        video_id = analyzed_video.get("video_id") if analyzed_video else None
        video_title = analyzed_video.get("title") if analyzed_video else None

        return IngestionResult(
            source_type=target_info["type"],
            source_name=channel_data["title"],
            channel_id=channel_data["channel_id"],
            channel_title=channel_data["title"],
            thumbnail_url=channel_data.get("thumbnail_url"),
            custom_url=custom_url,
            channel_url=channel_url,
            description=channel_data.get("description"),
            video_id=video_id,
            video_title=video_title,
            analyzed_video=analyzed_video,
            recent_videos=channel_data.get("recent_videos", []),
            video_count=channel_data.get("video_count", 1),
            metrics_data={
                "subscribers": channel_data.get("subscriber_count", 0),
                "total_views": channel_data.get("view_count", 0),
                "videos": channel_data.get("video_count", 1),
                "ctr": None,
                "avg_retention_percent": None,
            },
            coverage_status="complete",
            provenance_signature=f"sha256_{abs(hash(target)):x}",
        )

    def process_file_upload(self, file_path: Path) -> IngestionResult:
        if not file_path.exists():
            raise InvalidInputException(f"File not found: {file_path.name}")

        stem = file_path.stem
        # Strip internal uuid hex prefix if present
        if len(stem) > 33 and stem[32] == "_":
            clean_display_name = stem[33:]
        else:
            clean_display_name = stem

        clean_channel_title = clean_display_name.replace("_", " ").replace("-", " ").title()

        suffix = file_path.suffix.lower()
        content_bytes = file_path.read_bytes()

        if suffix == ".json":
            try:
                data = json.loads(content_bytes.decode("utf-8"))
                return IngestionResult(
                    source_type="json_analytics",
                    source_name=clean_display_name + suffix,
                    channel_title=data.get("channel_title", clean_channel_title),
                    metrics_data=data.get("metrics", data),
                    coverage_status="validated",
                    provenance_signature=f"file_{clean_display_name}",
                )
            except Exception as e:
                raise InvalidInputException(f"Malformed JSON file: {e}")

        elif suffix == ".csv":
            try:
                text = content_bytes.decode("utf-8", errors="replace")
                reader = csv.DictReader(io.StringIO(text))
                rows = list(reader)
                return IngestionResult(
                    source_type="csv_analytics",
                    source_name=clean_display_name + suffix,
                    channel_title=clean_channel_title,
                    video_count=len(rows),
                    metrics_data={"rows_imported": len(rows), "columns": reader.fieldnames or []},
                    coverage_status="validated",
                    provenance_signature=f"file_{clean_display_name}",
                )
            except Exception as e:
                raise InvalidInputException(f"Malformed CSV file: {e}")

        elif suffix in [".txt", ".vtt", ".srt"]:
            text = content_bytes.decode("utf-8", errors="replace")
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            return IngestionResult(
                source_type="transcript_file",
                source_name=file_path.name,
                channel_title=file_path.stem.title(),
                transcripts=[{"line_count": len(lines), "sample": lines[:5]}],
                coverage_status="validated",
                provenance_signature=f"file_{file_path.name}",
            )

        raise InvalidInputException(f"Unsupported file format: {suffix}. Allowed: .csv, .json, .txt, .srt, .vtt")
