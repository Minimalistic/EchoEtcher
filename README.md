# EchoEtcher

An automated system that converts audio notes into formatted Markdown notes with allowed tags using local AI processing.

## Project Status ⚠️

**Experimental / Work in Progress**

This project is a rapid prototype developed through collaborative AI assistance and personal iteration. As such, it comes with some important caveats:

- 🧪 Experimental: The codebase is in active development
- 🛠 Unstable: Expect potential breaking changes
- 🐛 Limited Testing: Minimal comprehensive testing has been performed
- 🚧 Use at Your Own Risk: Not recommended for production environments without significant review and modification

Contributions, feedback, and improvements are welcome! If you encounter issues or have suggestions, please open an issue on the repository.

## Disclaimer

By using EchoEtcher, you acknowledge that you understand and accept the risks associated with using an experimental project. You agree to hold harmless the developers and contributors of EchoEtcher for any damages or losses resulting from its use.

## Features

- **Folder Monitoring**: Monitors a selected folder (e.g., iCloud) for new audio files
- **Local Transcription**: Transcribes audio locally using Whisper (configurable model size)
- **Large File Support**: Automatically handles large audio files by chunking them into manageable segments
- **AI Processing**: Processes transcriptions locally with Ollama
- **Smart Note Generation**: Auto-generates formatted Markdown notes with allowed tags
- **Audio Embedding**: Links original audio files in Obsidian vault
- **iCloud Sync Support**: Handles iCloud file synchronization gracefully
- **Error Handling**: Robust error handling with retry logic and error directory
- **Processing Tracking**: Tracks processing time and provides detailed logging
- **Configuration Validation**: Validates configuration on startup with helpful error messages
- **CLI Options**: Dry-run mode, verbose logging, and version information
- **Completely Local**: No cloud services required - all processing happens on your machine

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed locally
- FFmpeg installed for audio processing
  - **macOS**: `brew install ffmpeg`
  - **Linux**: `sudo apt install ffmpeg` (Debian/Ubuntu) or `sudo yum install ffmpeg` (RHEL/CentOS)
  - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use `choco install ffmpeg`
- Obsidian vault set up

### Hardware Acceleration

EchoEtcher automatically detects and uses the best available hardware:

- **NVIDIA GPU (Windows/Linux)**: Automatically detects and uses CUDA if PyTorch with CUDA support is installed
- **CPU**: Uses CPU processing on macOS and when no GPU is available
  - **Note**: MPS (Apple Silicon GPU) is not used due to compatibility issues with Whisper's sparse tensor operations. CPU processing works reliably on macOS.

The device being used is logged at startup. No configuration needed - it just works!

## File Processing and Storage

### Notes Folder Structure

The `Daily Notes` folder is used for processed files with the following conventions:

- A sub-folder `audio` is created to store processed audio files
- Markdown files are generated for each processed audio file
- A markdown file is saved in the `Daily Notes` folder for each processed audio file and follows the format: `yyyy-MM-dd-Succinct-Generated-File-Name.md`

### Tag Processing

When processing tags from the transcribed audio, EchoEtcher uses a unique approach to tag management:

- The system allows you to specify a set of `allowed_tags`
- When writing the final tags to the Markdown file, these tags are **prepended** with '#echo-etcher/' 
- Prepending makes it easier to filter and search for specific tags that were created by EchoEtcher in your notes
- This approach ensures consistency and improves note organization and discoverability

## Setup

1. Clone this repository
2. (Optional) Create and activate a virtual environment:
   ```bash
   # Using venv (recommended for most users)
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   
   # Alternative: Use conda or your preferred environment manager
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and configure your paths
5. With Ollama.ai installed locally, open a new terminal and run `ollama pull mistral` (or whatever model you want to use)
6. Ensure the Ollama model you want to use is set in the `.env` file (By default the example .env is set to `mistral`)
7. Ensure Ollama is running locally by running `ollama serve`
8. Run the watcher:
   ```bash
   python main.py
   ```

### Command Line Options

EchoEtcher supports several command-line options:

- `--dry-run`: Test mode - scans and logs files but doesn't process them
- `--verbose` or `-v`: Enable debug/verbose logging
- `--version`: Show version information

Examples:
```bash
# Normal operation
python main.py

# Test configuration without processing files
python main.py --dry-run

# Enable detailed logging
python main.py --verbose
```
## Configuration

Edit the `.env` file to configure:

### Required Settings
- `WATCH_FOLDER`: Path to folder to watch for audio files (e.g., iCloud folder)
- `OBSIDIAN_VAULT_PATH`: Path to your Obsidian vault root directory
- `OLLAMA_MODEL`: Ollama model name to use (e.g., `mistral`, `llama2`)

### Optional Settings
- `NOTES_FOLDER`: Folder name within vault where notes are created (default: `notes`)
- `OLLAMA_API_URL`: Ollama API endpoint (default: `http://localhost:11434/api/generate`)
- `OLLAMA_TEMPERATURE`: Temperature for AI processing, 0.0-1.0 (default: `0.3`)
- `OLLAMA_TIMEOUT`: API timeout in seconds (default: `120`)
- `WHISPER_MODEL_SIZE`: Whisper model size - `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3` (default: `medium`)
- `SEQUENTIAL_PROCESSING`: Enable sequential processing mode - unloads Whisper after transcription to free memory (default: `true`)
- `ALLOWED_TAGS_FILE`: Path to allowed tags file (default: `allowed_tags.md`)
- `WHISPER_CHUNK_THRESHOLD`: Duration in seconds above which files are automatically chunked (default: `240` = 4 minutes)
- `WHISPER_CHUNK_DURATION`: Duration in seconds for each chunk when processing large files (default: `30`)
- `WHISPER_CHUNK_OVERLAP`: Overlap in seconds between chunks to ensure smooth merging (default: `5`)

### Large File Handling

EchoEtcher automatically handles large audio files by splitting them into smaller chunks for processing. This prevents memory issues and processing failures that can occur with very long recordings.

**How it works:**
- Files longer than 4 minutes (240 seconds) are automatically split into 30-second chunks with 5-second overlaps
- Each chunk is transcribed separately, then intelligently merged back together
- The overlap ensures smooth transitions between chunks and prevents missing words at boundaries
- All chunking happens automatically - no manual intervention needed

**Configuration:**
You can customize the chunking behavior via environment variables:
- `WHISPER_CHUNK_THRESHOLD`: Files longer than this (in seconds) will be chunked (default: 240)
- `WHISPER_CHUNK_DURATION`: Size of each chunk in seconds (default: 30)
- `WHISPER_CHUNK_OVERLAP`: Overlap between chunks in seconds (default: 5)

**Benefits:**
- ✅ No more 4-minute recording limits - process files of any length
- ✅ Prevents memory errors and crashes
- ✅ More reliable processing of long recordings
- ✅ Automatic fallback to chunking if single-file processing fails

### Hardware Recommendations

EchoEtcher uses **sequential processing by default** to optimize memory usage. This means Whisper unloads after transcription, freeing memory for Ollama processing. This allows you to use larger, higher-quality models even on systems with limited VRAM/RAM.

**Large File Processing:**
The chunking system means you can process files of any length, even on systems with limited memory. The default 30-second chunks are small enough to process reliably on most systems.

#### Recommended Configurations

**For systems with 8GB VRAM (e.g., RTX 3060 Ti):**
```bash
WHISPER_MODEL_SIZE=medium
OLLAMA_MODEL=mistral
SEQUENTIAL_PROCESSING=true
```
This configuration uses Whisper `medium` (~5GB) and Mistral 7B (~4-5GB) sequentially, fitting comfortably within 8GB VRAM.

**For systems with 16GB+ shared memory (e.g., MacBook Air M4):**
```bash
WHISPER_MODEL_SIZE=medium
OLLAMA_MODEL=mistral
SEQUENTIAL_PROCESSING=true
```
The same configuration works well on systems with more memory, providing excellent quality while maintaining efficient resource usage.

**For systems with limited memory (<8GB):**
```bash
WHISPER_MODEL_SIZE=small
OLLAMA_MODEL=phi-2
SEQUENTIAL_PROCESSING=true
```

**For maximum quality (requires 16GB+ VRAM/RAM):**
```bash
WHISPER_MODEL_SIZE=large-v3
OLLAMA_MODEL=mistral
SEQUENTIAL_PROCESSING=true
```

#### Model Size Reference

**Whisper Models:**
- `tiny` (~1GB): Fast, basic quality
- `base` (~1GB): Fast, good quality
- `small` (~2GB): Fast, very good quality
- `medium` (~5GB): Moderate speed, excellent quality ⭐ **Recommended**
- `large-v3` (~10GB): Slower, best quality

**Ollama Models:**
- `tinyllama` (~2GB): Fast, basic quality
- `phi-2` (~2GB): Fast, good quality
- `mistral` (~4-5GB): Moderate speed, very good quality ⭐ **Recommended**
- `llama2:7b` (~4-5GB): Moderate speed, very good quality

The application will validate your configuration on startup and provide helpful error messages if anything is missing or incorrect.

### Allowed Tags File

The `allowed_tags.md` file defines the list of tags that can be used in your notes. This helps maintain consistency and organization in your note-taking system.

#### Location
By default, the allowed tags file should be placed in your Obsidian vault's root directory. The location is specified by the `ALLOWED_TAGS_FILE` setting in the `.env` file.

#### File Format
Create a Markdown file with a list of allowed tags, one per line. Each tag should:
- Start with a `#`
- Use lowercase letters
- Use hyphens for multi-word tags
- Optionally use hierarchical tags with `/`

Example `allowed_tags.md`:
```markdown
# Allowed Tags

#meeting
#idea
#project
#journal
#family
#biking
#bills
#books/fiction
#books/non-fiction
#health
#finance
#quote
#career
#gaming
#rant
```

#### Purpose
- Restricts tags to a predefined list to prevent hallucinations
- Ensures consistent tag usage across notes
- Helps with organization and searchability
- Can be easily modified as your note-taking needs evolve
