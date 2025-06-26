# NarrateLM

A Python CLI tool that converts EPUB books to audio using Google's Gemini Text-to-Speech API.

## Features

- Extract chapters from EPUB files
- Convert text to high-quality speech using Gemini TTS
- Automatically split long chapters into manageable chunks
- Clean HTML content and format text for optimal speech synthesis
- Save audio files with descriptive names

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd narratelm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install as a package:
```bash
pip install -e .
```

## Usage

### Environment Setup

You need a Gemini API key to use this tool. You can:

1. Set it as an environment variable:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

2. Or pass it directly with the `--api-key` option

### Basic Usage

```bash
python narratelm.py /path/to/book.epub /path/to/output/directory
```

Or if installed as a package:
```bash
narratelm /path/to/book.epub /path/to/output/directory
```

### Options

- `--api-key`: Your Gemini API key (can also be set via `GEMINI_API_KEY` environment variable)
- `--max-chars`: Maximum characters per TTS request (default: 4000)

### Example

```bash
# Using environment variable for API key
export GEMINI_API_KEY="your-api-key"
python narratelm.py "my-book.epub" "./audiobook_output"

# Using command line option for API key
python narratelm.py "my-book.epub" "./audiobook_output" --api-key "your-api-key"

# Adjusting chunk size for longer requests
python narratelm.py "my-book.epub" "./audiobook_output" --max-chars 5000
```

## Output

The tool will create WAV audio files in the specified output directory with names like:
- `chapter_01_Introduction.wav`
- `chapter_02_Chapter_Title.wav` 
- `chapter_03_Long_Chapter_part_01.wav` (for chapters split into multiple parts)

## Requirements

- Python 3.8+
- Gemini API key
- Dependencies listed in `requirements.txt`

## Limitations

- Only supports EPUB format
- Requires internet connection for TTS API calls
- API usage costs apply (check Gemini API pricing)
- Large books may take considerable time to process

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License - see LICENSE file for details.