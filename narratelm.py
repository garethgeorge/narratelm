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
import concurrent.futures
import os


# --- Constants ---
TTS_MODEL = "gemini-2.5-flash-preview-tts"
# TTS_VOICE = "Umbrial"
# TTS_VOICE = "Iapetus"
TTS_VOICE = "Enceladus"
AUDIO_BITRATE = "96k"
TTS_PROMPT = (
    "The following text is extracted from an epub, please read it aloud as an "
    "audiobook narrator announcing chapter titles as you see them and providing "
    "different intonations for different characters:\n\n"
)
MIN_CHAPTER_LENGTH = 100
DEFAULT_MAX_CHARS = 16000


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

    chapter_items = []

    if book.spine:
        click.echo(f"Found and using book.spine: {book.spine}")
        chapters_in_order = []
        for item_id, _ in book.spine:
            item = book.get_item_with_id(item_id)
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                chapter_items.append(item)
    else:
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            chapter_items.append(item)

    chapters = []
    chapter_num = 1
    for item in chapter_items:
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


def get_audio_duration(file_path: str) -> float:
    """Get the duration of an audio file in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                file_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except FileNotFoundError:
        click.echo("Error: ffprobe is not installed or not found in PATH", err=True)
        return 0.0
    except (subprocess.CalledProcessError, ValueError) as e:
        click.echo(f"Error getting duration for {file_path}: {e}", err=True)
        return 0.0


def extract_cover_image(book: epub.EpubBook, output_dir: Path) -> Optional[Path]:
    """Extracts the cover image from the EPUB and saves it to a file."""
    cover_item = None

    # 1. Try to find cover image from metadata (most reliable for EPUB2/3)
    try:
        cover_meta = book.get_metadata("OPF", "cover")
        if cover_meta:
            cover_id = cover_meta[0][1].get("content")
            if cover_id:
                cover_item = book.get_item_with_id(cover_id)
    except (IndexError, KeyError):
        pass  # Metadata not found or malformed

    # 2. If not found, look for an item with type ITEM_COVER
    if not cover_item:
        covers = list(book.get_items_of_type(ebooklib.ITEM_COVER))
        if covers:
            cover_item = covers[0]

    # 3. Fallback: Look for common filenames in all image items
    if not cover_item:
        for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
            if "cover" in item.get_name().lower():
                cover_item = item
                break

    if cover_item:
        cover_data = cover_item.get_content()
        cover_ext = Path(cover_item.get_name()).suffix or ".jpg"
        cover_path = output_dir / f"cover{cover_ext}"
        with open(cover_path, "wb") as f:
            f.write(cover_data)
        click.echo(f"  \u2713 Extracted cover image to {cover_path}")
        return cover_path

    click.echo("  \u2718 Could not find a cover image in the EPUB.")
    return None


def merge_audio_files(
    m4a_files: List[str],
    output_path: Path,
    epub_path: Path,
    chapters: List[Tuple[str, str]],
):
    """
    Merges M4A files into a single file with chapters and cover art.
    """
    click.echo("Starting final merge process...")

    # 1. Create file list for ffmpeg concat
    file_list_path = output_path.parent / "ffmpeg_file_list.txt"
    with open(file_list_path, "w") as f:
        for m4a_file in m4a_files:
            # Use absolute paths in the file list for safety
            f.write(f"file '{os.path.abspath(m4a_file)}'\n")
    click.echo(f"  \u2713 Created ffmpeg file list: {file_list_path}")

    # 2. Extract cover art
    book = epub.read_epub(epub_path)
    cover_image_path = extract_cover_image(book, output_path.parent)

    # 3. Generate chapter metadata
    metadata_path = output_path.parent / "ffmpeg_metadata.txt"
    chapter_files = {}
    for m4a_file in m4a_files:
        match = re.match(r"(\d+)_", Path(m4a_file).name)
        if match:
            chap_num = int(match.group(1))
            if chap_num not in chapter_files:
                chapter_files[chap_num] = []
            chapter_files[chap_num].append(m4a_file)

    with open(metadata_path, "w") as f:
        f.write(";FFMETADATA1\n")
        current_start_time_ms = 0
        for chap_num in sorted(chapter_files.keys()):
            if chap_num - 1 < len(chapters):
                chapter_title = chapters[chap_num - 1][0].replace("_", " ")
            else:
                chapter_title = f"Chapter {chap_num}"

            chapter_duration_ms = 0
            for m4a_file in chapter_files[chap_num]:
                duration = get_audio_duration(m4a_file)
                chapter_duration_ms += int(duration * 1000)

            end_time_ms = current_start_time_ms + chapter_duration_ms
            f.write("[CHAPTER]\n")
            f.write("TIMEBASE=1/1000\n")
            f.write(f"START={current_start_time_ms}\n")
            f.write(f"END={end_time_ms}\n")
            f.write(f"title={chapter_title}\n")
            current_start_time_ms = end_time_ms
    click.echo(f"  \u2713 Created ffmpeg metadata file: {metadata_path}")

    # 4. Run ffmpeg command
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(file_list_path),
        "-i",
        str(metadata_path),
    ]
    if cover_image_path:
        ffmpeg_cmd.extend(["-i", str(cover_image_path)])
        ffmpeg_cmd.extend(["-map", "0:a", "-map_metadata", "1", "-map", "2:v"])
        ffmpeg_cmd.extend(["-c:v", "copy", "-disposition:v:0", "attached_pic"])
    else:
        ffmpeg_cmd.extend(["-map", "0:a", "-map_metadata", "1"])

    ffmpeg_cmd.extend(["-c:a", "copy", str(output_path)])

    click.echo(f"  Running ffmpeg to merge files...")
    try:
        subprocess.run(
            ffmpeg_cmd,
            check=True,
            capture_output=True,
        )
        click.echo(f"  \u2713 Successfully merged to {output_path}")
    except FileNotFoundError:
        click.echo("Error: ffmpeg is not installed or not found in PATH", err=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"Error during merge: {e.stderr.decode()}", err=True)
    finally:
        # 5. Clean up temporary files
        file_list_path.unlink()
        metadata_path.unlink()
        if cover_image_path and cover_image_path.exists():
            cover_image_path.unlink()
        click.echo("  \u2713 Cleaned up temporary files.")


def convert_wav_to_m4a(wav_path: str, m4a_path: str):
    """Convert a WAV file to M4A format using ffmpeg."""
    try:
        subprocess.run(
            ["ffmpeg", "-i", wav_path, "-c:a", "aac", "-b:a", AUDIO_BITRATE, m4a_path],
            check=True,
            capture_output=True,
        )
        click.echo(f"  \u2713 Converted to M4A: {m4a_path}")
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
) -> Tuple[int, List[str]]:
    """Processes a single chapter, generating and saving audio files."""
    click.echo(f"Processing chapter {chapter_index}/{total_chapters}: {chapter_title}")

    text_chunks = split_text_into_chunks(chapter_text, max_chars)
    click.echo(f"  Split into {len(text_chunks)} chunk(s) for TTS processing")

    total_tokens = 0
    processed_files = []
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
            processed_files.append(str(m4a_path))
            continue

        click.echo(f"  Generating audio for {filename}...")
        speech_result = generate_speech(client, chunk_text)

        if not speech_result:
            click.echo(f"  \u2718 Failed to generate audio for {filename}", err=True)
            continue

        audio_data, usage_metadata = speech_result
        wave_file(str(output_path), audio_data)
        click.echo(f"  \u2713 Saved: {output_path}")

        convert_wav_to_m4a(str(output_path), str(m4a_path))
        processed_files.append(str(m4a_path))
        output_path.unlink()  # Remove intermediate WAV file

        if usage_metadata:
            chunk_tokens = usage_metadata.total_token_count
            total_tokens += chunk_tokens
            click.echo(f"  Tokens for audio chunk: {chunk_tokens}")

    return total_tokens, processed_files


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

    print(f"Submitting {total_chapters} chapters to the thread pool...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for i, (chapter_title, chapter_text) in enumerate(chapters, 1):
            future = executor.submit(
                process_chapter,  # The function to execute
                # Arguments for the process_chapter function
                chapter_index=i,
                chapter_title=chapter_title,
                chapter_text=chapter_text,
                total_chapters=total_chapters,
                client=client,
                output_dir=output_dir,
                max_chars=max_chars,
            )
            futures.append(future)

        click.echo("Waiting for chapters to be processed...")
        total_tokens_used = 0
        all_m4a_files = []
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            completed_count += 1
            try:
                tokens_for_chapter, m4a_files_for_chapter = future.result()
                total_tokens_used += tokens_for_chapter
                all_m4a_files.extend(m4a_files_for_chapter)
                click.echo(
                    f"({completed_count}/{total_chapters}) Chapter processed. Tokens used so far: {total_tokens_used}"
                )
            except Exception as exc:
                click.echo(f"A chapter processing task generated an exception: {exc}")

    all_m4a_files.sort()

    if all_m4a_files:
        final_m4a_path = output_dir / f"{epub_path.stem}.m4a"
        merge_audio_files(all_m4a_files, final_m4a_path, epub_path, chapters)
        click.echo(f"\u2705 Completed! Final audio file saved to: {final_m4a_path}")
    else:
        click.echo("No audio files were generated, skipping merge.")

    click.echo(f"Total tokens used for the entire book: {total_tokens_used}")


if __name__ == "__main__":
    main()
