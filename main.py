import time
import argparse
from youtube_transcript_api._api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import re
from loguru import logger
import subprocess
from sqlmodel import SQLModel, Field, Session, select, create_engine
import datetime
import os

PROJECT_BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_BASE_PATH, "data/transcripts.db")
SUMMARIZATION_PROMPT = """**Persona:** You are an Expert Video Content Analyst and Summarizer. Your expertise lies in distilling complex video information into clear, concise, and actionable summaries.

**Primary Goal:** To create a comprehensive yet digestible summary of the provided YouTube video. This summary must enable a user to thoroughly understand the video's core message, key announcements, significant data points, strategic implications, and any calls to action, effectively replacing the need for them to watch the video itself.

**Input:** You will be provided with a YouTube video URL. You are expected to process its content (transcript, and if possible, infer visual/audio cues).

**Output Requirements:**

1.  **Language:** Strictly {summary_language}. (applies to all summaries).
2.  **Tone:** Professional, objective, and informative (applies to all summaries).
3.  **Multi-Part Summary Output:** The entire output should consist of four distinct summaries, separated by "---".

    **Part 1: Full Comprehensive Summary**
    *   **Executive Summary:**
        *   Begin with a 2-4 sentence executive summary.
        *   This summary should incisively capture the video's central theme, primary objectives/announcements, and its overall significance or potential impact.
    *   **Key Takeaways List:**
        *   Following the executive summary, provide a bulleted list of key takeaways.
        *   **Prioritization Criteria:** Focus on extracting information that is:
            *   New Product/Feature Announcements: Clearly state what is new and its core functionality.
            *   Significant Data/Evidence: Include specific metrics, statistics, or crucial evidence presented.
            *   Core Arguments/Problems Addressed: Identify the main challenges or topics the video discusses.
            *   Solutions/Conclusions Offered: Detail the proposed solutions, findings, or conclusions.
            *   Strategic Insights: If the video discusses company strategy, market positioning, or future directions, capture these.
            *   Actionable Advice/Calls to Action (from the video): Note any specific recommendations or requests made to the audience by the video's speakers.
        *   **Formatting for Each Takeaway:**
            *   -[Timestamp URL] Emoji **[Concise & Descriptive Headline (aim for 3-8 words)]**: [Detailed Elaboration (1-3 sentences). Include crucial details, facts, names, dates, specific features, or outcomes. Be specific and avoid vagueness.]
            *   **Timestamp:** Markdown URL format (e.g., [0:35](URL&t=35s)). The URL text should be the timestamp.
            *   **Emoji:** Relevant and *unique* for each takeaway item. Do not use brackets around the emoji.
            *   **Headline:** Make it highly descriptive, using strong verbs where appropriate.
            *   **Elaboration:** Provide enough context and detail for the takeaway to be understood stand-alone. If an announcement has multiple sub-points, consider if they warrant separate takeaways or a more detailed elaboration.
        *   **Exclusions:**
            *   Do *not* prepend takeaways with "Key takeaway:".
            *   Omit generic introductions, filler content, repetitive pleasantries, off-topic discussions, and non-informative calls to like/subscribe/share (unless the *video itself* is about channel growth strategies, for example).
            *   Skip overt sponsor messages unless the sponsorship itself is a key point of the video (e.g., a new partnership announcement).

    ---

    **Part 2: Medium Summary (Approximately 1/2 the length/detail of the Part1 Key Takeaways List)**
    *   Provide a list of key takeaways, selecting only the *most critical* announcements, objectives, the core problem/solution discussed, and significant conclusions or implications from the Full Summary. The number of takeaways should be roughly half of those in the Full Summary's list.

    ---

    **Part 3: Short Summary (Approximately 1/2 the length/detail of the Part2 Key Takeaways List)**
    *   Provide a further condensed list of key takeaways, highlighting only the *absolute core message*, the primary experiment/topic, and its *key outcome or main point of discussion*. The number of points should be roughly half of those in the Medium Summary's list.

    ---

    **Part 4: Very Short Summary (Approximately 1/2 the length/detail of the Part3 Key Takeaways List)**
    *   Deliver a highly concise list of roughly half of those in the Short Summary's list. that capture the "elevator pitch" essence of the video (what it's about and its most significant finding/demonstration).

**Video URL:**
{youtube_video_url}

**TimeStamped Video Transcript:**
{video_transcript}
"""
# youtube_video_url = "https://www.youtube.com/watch?v=FCE_JyeJzJg&list=PLFr7f4WLNwrZzhz-YDjha6j3Z9ymjo7rD&index=3"


def extract_video_id(url: str) -> str:
    """
    Extracts the video ID from a YouTube URL.
    Raises a ValueError if unable to extract the video ID.
    """
    match = re.search(r"v=([\w-]+)", url)
    if match:
        return match.group(1)
    raise ValueError("[Error] Could not extract video ID from URL.")


def convert_seconds_to_timestamp(seconds: float) -> str:
    """
    Converts a time duration in seconds to a timestamp string (HH:MM:SS),
    with each part zero-padded to 2 digits.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    # Use the :02 format specifier to pad each part with a leading zero if needed
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def get_video_transcript(video_id: str, proxy: str | None = None) -> tuple[str, str]:
    """
    Retrieves the transcript for the YouTube video specified by youtube_video_url.
    Returns a tuple (transcript, transcription_language), or raises an error message if unable to do so.
    Implements an improved retry strategy with exponential backoff and logs all attempts.
    Optionally uses a proxy for network requests.
    """
    transcript_text = ""
    languages = ("en", "fa")
    max_retries = 7
    base_delay = 2  # seconds
    for language in languages:
        logger.info(f"Trying language: {language}")
        for n_try in range(1, max_retries + 1):
            try:
                logger.debug(f"Attempt {n_try} for language '{language}'")
                # Use proxy if provided, otherwise default
                transcript_languages = YouTubeTranscriptApi.list_transcripts(
                    video_id,
                    proxies={"http": proxy, "https": proxy} if proxy else None,
                )
                logger.debug(transcript_languages)
                transcript = transcript_languages.find_transcript([language])
                logger.debug(f"Found transcript: {transcript}")
                fetched_transcript = transcript.fetch().to_raw_data()
                transcript_text = "\n".join(
                    [
                        f"[{convert_seconds_to_timestamp(entry['start'])}]{entry['text']}"
                        for entry in fetched_transcript
                    ]
                )
                return transcript_text, language
            except TranscriptsDisabled:
                logger.error("[Error] Transcripts are disabled for this video.")
                break
            except NoTranscriptFound:
                logger.error(
                    f"[Error] No transcript found for this video in the {language} language."
                )
                break
            except Exception as e:
                logger.error(f"[Error] Attempt {n_try} failed: {e}")
                if n_try < max_retries:
                    delay = base_delay * (2 ** (n_try - 1))
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error("Max retries reached. Giving up.")
    raise ValueError("[Error] Could not retrieve transcript after multiple attempts.")


class Transcript(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    video_id: str
    url: str
    transcript: str
    transcription_language: str
    created_at: str


def get_engine():
    return create_engine(f"sqlite:///{DB_PATH}", echo=False)


def init_db():
    """
    Initializes the SQLite database and creates the transcripts table if it doesn't exist.
    """
    engine = get_engine()
    SQLModel.metadata.create_all(engine)


def store_transcript(
    video_id: str,
    url: str,
    transcript: str,
    transcription_language: str,
):
    """
    Stores the transcript in the SQLite database using SQLModel ORM.
    """
    engine = get_engine()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    transcript_obj = Transcript(
        video_id=video_id,
        url=url,
        transcript=transcript,
        transcription_language=transcription_language,
        created_at=now,
    )
    with Session(engine) as session:
        session.add(transcript_obj)
        session.commit()


def get_transcript_from_db(video_id: str):
    """
    Retrieves the transcript and transcription_language from the DB for the given video_id.
    Returns (transcript, transcription_language) or (None, None) if not found.
    """
    engine = get_engine()
    with Session(engine) as session:
        statement = (
            select(Transcript)
            .where(Transcript.video_id == video_id)
            .order_by(Transcript.created_at.desc())
            .limit(1)
        )
        result = session.exec(statement).first()
        if result:
            return result.transcript, result.transcription_language
        return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a YouTube video by URL.")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="Bypass DB and fetch transcript from YouTube",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="Proxy URL (e.g., http://127.0.0.1:8086 or socks5h://127.0.0.1:2080)",
    )
    args = parser.parse_args()
    youtube_video_url = args.url
    force_fetch = args.force_fetch
    proxy = args.proxy

    init_db()  # Ensure DB and table exist

    try:
        video_id = extract_video_id(youtube_video_url)
        video_transcript = None
        transcription_language = None

        if not force_fetch:
            video_transcript, transcription_language = get_transcript_from_db(video_id)
            if video_transcript:
                logger.info("Transcript loaded from DB.")
        if not video_transcript:
            video_transcript, transcription_language = get_video_transcript(
                video_id, proxy=proxy
            )
            try:
                store_transcript(
                    video_id,
                    youtube_video_url,
                    video_transcript,
                    transcription_language,
                )
                logger.info("Transcript stored in DB.")
            except Exception as e:
                logger.error(f"Failed to store transcript in DB: {e}")

    except ValueError as e:
        logger.error(f"[Error] {e}")
        return

    final_prompt = SUMMARIZATION_PROMPT.format(
        youtube_video_url=f"https://www.youtube.com/watch?v={video_id}",
        video_transcript=video_transcript,
        summary_language="Original Language of the video",
    )
    if len(final_prompt) > 2000:
        logger.debug(final_prompt[:1000] + "\n...\n" + final_prompt[-1000:])
    else:
        logger.debug(final_prompt)
    try:
        subprocess.run("pbcopy", text=True, input=final_prompt, check=True)
        logger.info("Final prompt copied to clipboard.")
    except Exception as e:
        logger.error(f"Failed to copy to clipboard: {e}")
    transcribe_word_count = len(video_transcript.split())
    logger.debug(f"Transcribe Word Count: {transcribe_word_count:,}")
    final_prompt_word_count = len(final_prompt.split())
    logger.debug(f"Final Prompt Word Count: {final_prompt_word_count:,}")


if __name__ == "__main__":
    main()
