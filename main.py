import time
from youtube_transcript_api._api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import re
from loguru import logger

SUMMARIZATION_PROMPT = """**Persona:** You are an Expert Video Content Analyst and Summarizer. Your expertise lies in distilling complex video information into clear, concise, and actionable summaries.

**Primary Goal:** To create a comprehensive yet digestible summary of the provided YouTube video. This summary must enable a user to thoroughly understand the video's core message, key announcements, significant data points, strategic implications, and any calls to action, effectively replacing the need for them to watch the video itself.

**Input:** You will be provided with a YouTube video URL. You are expected to process its content (transcript, and if possible, infer visual/audio cues).

**Output Requirements:**

1.  **Language:** Strictly Original Language of the video. (applies to all summaries).
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
# youtube_video_url = "https://www.youtube.com/watch?v=VEByHg_aFPI"
youtube_video_url = "https://www.youtube.com/watch?v=FCE_JyeJzJg&list=PLFr7f4WLNwrZzhz-YDjha6j3Z9ymjo7rD&index=3"


def extract_video_id(url: str) -> str | None:
    """
    Extracts the video ID from a YouTube URL.
    """
    match = re.search(r"v=([\w-]+)", url)
    return match.group(1) if match else None


def convert_seconds_to_timestamp(seconds: float) -> str:
    """
    Converts a time duration in seconds to a timestamp string (HH:MM:SS).
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}:{minutes}:{seconds}"


def get_video_transcript() -> str:
    """
    Retrieves the transcript for the YouTube video specified by youtube_video_url.
    Returns the transcript as a single string, or raises an error message if unable to do so.
    Implements an improved retry strategy with exponential backoff and logs all attempts.
    """
    video_id = extract_video_id(youtube_video_url)
    if not video_id:
        return "[Error] Could not extract video ID from URL."
    transcript_text = ""
    languages = ("en", "fa")
    max_retries = 7
    base_delay = 2  # seconds
    for language in languages:
        logger.info(f"Trying language: {language}")
        for n_try in range(1, max_retries + 1):
            try:
                logger.debug(f"Attempt {n_try} for language '{language}'")
                transcript_languages = YouTubeTranscriptApi.list_transcripts(video_id)
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
                return transcript_text
            except TranscriptsDisabled:
                logger.error("[Error] Transcripts are disabled for this video.")
                break
            except NoTranscriptFound:
                logger.error("[Error] No transcript found for this video.")
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


def main() -> None:
    try:
        video_transcript = get_video_transcript()
    except ValueError as e:
        logger.error(f"[Error] {e}")
        return

    final_prompt = SUMMARIZATION_PROMPT.format(
        youtube_video_url=youtube_video_url, video_transcript=video_transcript
    )
    print(final_prompt)


if __name__ == "__main__":
    main()
