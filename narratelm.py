#!/usr/bin/env python3
"""
NarrateLM - A CLI tool to narrate EPUB books using Gemini TTS API
"""

import sys
import re
import wave
from pathlib import Path
from typing import List, Tuple, Optional

import click
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
import subprocess

# --- Constants ---
TTS_MODEL = "gemini-2.5-flash-preview-tts"
TTS_VOICE = "Enceladus"
AUDIO_BITRATE = "96k"
TTS_PROMPT = (
    "The following text is extracted from an epub, please read it aloud as an "
    "audiobook narrator announcing chapter titles as you see them and providing "
    "different intonations for different characters:\n\n"
)
MIN_CHAPTER_LENGTH = 100
DEFAULT_MAX_CHARS = 8000


# --- Helper Functions ---


def wave_file(
    filename: str,
    pcm: bytes,
    channels: int = 1,
    rate: int = 24000,
    sample_width: int = 2,
):
    """Save PCM data to a WAV file."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def clean_text_for_tts(text: str) -> str:
    """Clean and prepare text for TTS conversion."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_html(html_content: bytes) -> str:
    """Extract clean text from HTML content."""
    soup = BeautifulSoup(html_content, "html.parser")
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text()
    return clean_text_for_tts(text)


def _clean_filename(name: str) -> str:
    """Clean a string to be a valid filename."""
    clean_name = re.sub(r"[^\w\s-]", "", name)
    clean_name = re.sub(r"[-\s]+", "_", clean_name)
    return clean_name


def _get_chapter_title(item: epub.EpubHtml, default_prefix: str) -> str:
    """Extract a title from an EPUB item, or create a default one."""
    if item.get_name() and not item.get_name().endswith((".xhtml", ".html")):
        return item.get_name()

    soup = BeautifulSoup(item.get_content(), "html.parser")
    title_tag = soup.find(["h1", "h2", "title"])
    if title_tag and title_tag.get_text().strip():
        return title_tag.get_text().strip()

    return default_prefix


def get_chapters_from_epub(epub_path: str) -> List[Tuple[str, str]]:
    """
    Extract chapters from an EPUB file.
    Returns a list of tuples: (chapter_title, chapter_text)
    """
    try:
        book = epub.read_epub(epub_path)
    except Exception as e:
        click.echo(f"Error reading EPUB file: {e}", err=True)
        sys.exit(1)

    chapters = []
    chapter_num = 1
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        text_content = extract_text_from_html(item.get_content())

        if len(text_content) < MIN_CHAPTER_LENGTH:
            continue

        default_title = f"Chapter {chapter_num}"
        chapter_title = _get_chapter_title(item, default_title)
        clean_title = _clean_filename(chapter_title)

        chapters.append((clean_title, text_content))
        chapter_num += 1

    return chapters


def generate_speech(
    client: genai.Client, text: str
) -> Optional[Tuple[bytes, types.UsageMetadata]]:
    """Generate speech from text using Gemini TTS API."""
    try:
        response = client.models.generate_content(
            model=TTS_MODEL,
            contents=TTS_PROMPT + text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=TTS_VOICE,
                        )
                    )
                ),
            ),
        )
        return (
            response.candidates[0].content.parts[0].inline_data.data,
            response.usage_metadata,
        )
    except Exception as e:
        click.echo(f"Error generating speech: {e}", err=True)
        return None


def split_text_into_chunks(text: str, max_chars: int) -> List[str]:
    """Split text into chunks of a maximum number of characters."""
    if len(text) <= max_chars:
        return [text]

    click.echo(
        f"Splitting text into chunks of max {max_chars} characters, size is {len(text)} characters"
    )
    chunks = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        # If adding this sentence exceeds the limit and we already have content
        if current_length + len(sentence) > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        # If a single sentence is longer than max_chars, we need to split it
        if len(sentence) > max_chars:
            # Add what we have so far
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            # Split long sentence by adding it directly as its own chunk
            chunks.append(sentence)
        else:
            # Add the sentence to the current chunk
            current_chunk.append(sentence)
            current_length += len(sentence) + 1  # +1 for space between sentences

    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks


def convert_wav_to_m4a(wav_path: str, m4a_path: str):
    """Convert a WAV file to M4A format using ffmpeg."""
    try:
        subprocess.run(
            ["ffmpeg", "-i", wav_path, "-c:a", "aac", "-b:a", AUDIO_BITRATE, m4a_path],
            check=True,
            capture_output=True,
        )
        click.echo(f"  ✓ Converted to M4A: {m4a_path}")
    except FileNotFoundError:
        click.echo("Error: ffmpeg is not installed or not found in PATH", err=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"Error during conversion: {e.stderr.decode()}", err=True)


def process_chapter(
    chapter_index: int,
    chapter_title: str,
    chapter_text: str,
    total_chapters: int,
    client: genai.Client,
    output_dir: Path,
    max_chars: int,
) -> int:
    """Processes a single chapter, generating and saving audio files."""
    click.echo(f"Processing chapter {chapter_index}/{total_chapters}: {chapter_title}")

    text_chunks = split_text_into_chunks(chapter_text, max_chars)
    click.echo(f"  Split into {len(text_chunks)} chunk(s) for TTS processing")

    total_tokens = 0
    for chunk_idx, chunk_text in enumerate(text_chunks):
        part_num = chunk_idx + 1
        if len(text_chunks) > 1:
            filename = f"{chapter_index:02d}_{chapter_title}_part_{part_num:02d}.wav"
        else:
            filename = f"{chapter_index:02d}_{chapter_title}.wav"

        output_path = output_dir / filename
        m4a_path = output_path.with_suffix(".m4a")

        if m4a_path.exists():
            click.echo(f"  File {m4a_path} already exists, skipping...")
            continue

        click.echo(f"  Generating audio for {filename}...")
        speech_result = generate_speech(client, chunk_text)

        if not speech_result:
            click.echo(f"  ✗ Failed to generate audio for {filename}", err=True)
            continue

        audio_data, usage_metadata = speech_result
        wave_file(str(output_path), audio_data)
        click.echo(f"  ✓ Saved: {output_path}")

        convert_wav_to_m4a(str(output_path), str(m4a_path))
        output_path.unlink()  # Remove intermediate WAV file

        if usage_metadata:
            chunk_tokens = usage_metadata.total_token_count
            total_tokens += chunk_tokens
            click.echo(f"  Tokens for audio chunk: {chunk_tokens}")

    return total_tokens


@click.command()
@click.argument("epub_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--api-key",
    envvar="GEMINI_API_KEY",
    required=True,
    help="Gemini API key (can also be set via GEMINI_API_KEY environment variable)",
)
@click.option(
    "--max-chars",
    default=DEFAULT_MAX_CHARS,
    help=f"Maximum characters per TTS request (default: {DEFAULT_MAX_CHARS})",
)
def main(epub_path: Path, output_dir: Path, api_key: str, max_chars: int):
    """
    Convert an EPUB book to audio using Gemini TTS API.

    EPUB_PATH: Path to the EPUB file to convert
    OUTPUT_DIR: Directory where audio files will be saved
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        click.echo(f"Error initializing Gemini client: {e}", err=True)
        sys.exit(1)

    click.echo(f"Reading EPUB file: {epub_path}")
    chapters = get_chapters_from_epub(str(epub_path))

    if not chapters:
        click.echo("No chapters found in the EPUB file.", err=True)
        sys.exit(1)

    total_chapters = len(chapters)
    click.echo(f"Found {total_chapters} chapters")

    total_tokens_used = 0
    for i, (chapter_title, chapter_text) in enumerate(chapters, 1):
        tokens_for_chapter = process_chapter(
            chapter_index=i,
            chapter_title=chapter_title,
            chapter_text=chapter_text,
            total_chapters=total_chapters,
            client=client,
            output_dir=output_dir,
            max_chars=max_chars,
        )
        total_tokens_used += tokens_for_chapter
        if tokens_for_chapter > 0:
            click.echo(f"  Total tokens used so far: {total_tokens_used}")

    click.echo(f"\nCompleted! Audio files saved to: {output_dir}")
    click.echo(f"Total tokens used for the entire book: {total_tokens_used}")


if __name__ == "__main__":
    main()
