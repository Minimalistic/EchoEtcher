from pathlib import Path
from typing import Dict, Optional, Any, Tuple
import logging
import yaml
import json

from .base_processor import BaseContentProcessor
from .audio_processor import AudioProcessor
from .text_processor import TextProcessor
from .image_processor import ImageProcessor
from .folder_processor import FolderProcessor

class ContentProcessorManager:
    def __init__(self, agent=None, max_formatting_retries: int = 3):
        self.agent = agent
        self.processors: Dict[str, BaseContentProcessor] = {}
        self._extension_map = {}
        self.max_formatting_retries = max_formatting_retries
        self._initialize_processors()
    
    def _initialize_processors(self):
        audio = AudioProcessor()
        text = TextProcessor()
        image = ImageProcessor()
        folder = FolderProcessor(content_processor_manager=self)
        
        self.processors['audio'] = audio
        self.processors['text'] = text
        self.processors['image'] = image
        self.processors['folder'] = folder
        
        for p in [audio, text, image]:
            for ext in p.get_supported_extensions():
                self._extension_map[ext.lower()] = p

    def get_processor(self, file_path: Path) -> Optional[BaseContentProcessor]:
        if file_path.is_dir():
            return self.processors['folder']
        return self._extension_map.get(file_path.suffix.lower())

    def can_process(self, file_path: Path) -> bool:
        if file_path.is_dir(): return True
        return file_path.suffix.lower() in self._extension_map

    async def _validate_formatted_content(self, raw_text: str, formatted_content: str) -> Tuple[bool, str]:
        """
        Validates that formatted content is appropriate compared to raw text.
        Uses a separate LLM call to check for issues.
        
        Returns:
            (is_valid, reason) - True if valid, False with reason if invalid
        """
        if not self.agent:
            return True, "No agent available for validation"
        
        # Quick length check first (heuristic)
        raw_length = len(raw_text.strip())
        formatted_length = len(formatted_content.strip())
        
        # If formatted is less than 30% of raw, likely too condensed
        if formatted_length < raw_length * 0.3:
            return False, f"Formatted content too short ({formatted_length} vs {raw_length} chars, {formatted_length/raw_length*100:.1f}%)"
        
        # If formatted is more than 150% of raw, likely added too much
        if formatted_length > raw_length * 1.5:
            return False, f"Formatted content too long ({formatted_length} vs {raw_length} chars, {formatted_length/raw_length*100:.1f}%)"
        
        # LLM validation for content quality
        validation_prompt = f"""
You are a quality validator for transcript formatting. Compare the RAW transcript with the FORMATTED version.

RAW TRANSCRIPT (original):
{raw_text[:5000]}

FORMATTED VERSION:
{formatted_content[:5000]}

Your task: Determine if the formatted version is appropriate.

VALID if:
- Contains the same key information and topics as raw
- Has proper formatting (paragraphs, lists, headers)
- Preserves the original meaning and details
- Only removes filler words and fixes punctuation
- Does NOT significantly condense or summarize

INVALID if:
- Missing major topics or information from raw
- Significantly condensed (less than 70% of original content)
- Added content not in original
- Changed meaning or facts
- Too much content removed

Respond in JSON format:
{{
  "valid": true or false,
  "reason": "Brief explanation of why it's valid or invalid",
  "issues": ["list of specific issues if invalid"]
}}
"""
        try:
            response_text = await self.agent.llm.generate_text(validation_prompt, json_mode=True)
            validation_result = json.loads(response_text)
            
            is_valid = validation_result.get('valid', False)
            reason = validation_result.get('reason', 'No reason provided')
            
            if not is_valid:
                issues = validation_result.get('issues', [])
                if issues:
                    reason = f"{reason}. Issues: {', '.join(issues)}"
            
            return is_valid, reason
            
        except Exception as e:
            # If validation fails, log but don't reject (fail open)
            await self.agent.log(f"Validation check failed: {e}, accepting formatted content")
            return True, f"Validation error (accepted): {e}"

    async def _format_with_llm(self, text: str, source_type: str, file_path: Path) -> Dict[str, Any]:
        """
        Internal method to format content with LLM. Returns the response dict.
        """
        prompt = f"""
You are a transcript formatter. Your job is to add STRUCTURE and FORMATTING to raw transcripts, NOT to rewrite them.

I have a {source_type} transcript from "{file_path.name}".

TRANSCRIPT:
{text[:10000]}

Your task:
1. Create a descriptive TITLE based on the main topics discussed.
2. Generate 3-5 relevant TAGS (e.g. #ideas, #meeting, #journal).
3. Write a 1-2 sentence SUMMARY in FIRST PERSON (use "I" not "The speaker").
4. Format the transcript following the rules below.

CRITICAL FORMATTING RULES:
- KEEP THE ORIGINAL WORDS EXACTLY. Do not paraphrase or rewrite sentences.
- Keep ALL profanity exactly as spoken - do not censor or remove any swear words.
- Only fix obvious transcription errors and punctuation.
- Keep all facts and details exactly as stated - do not change or misinterpret them.

MANDATORY FORMATTING REQUIREMENTS (YOU MUST APPLY THESE):
1. **Paragraph Breaks**: Add blank lines between distinct topics or thoughts. Each paragraph should be 2-4 sentences on a related topic.

2. **List Formatting**: When the speaker lists items (numbered or bulleted), format them as proper markdown lists:
   - Use `- ` for bullet points
   - Use `1. ` for numbered lists
   - Example: "First, speech-to-text. Second, AI processing. Third, UI design." becomes:
     - Speech-to-text
     - AI processing  
     - UI design

3. **Section Headers**: Add markdown headers (## Heading) when the topic clearly changes to a new subject. Use descriptive, concrete headers based on actual content (e.g. ## Meeting with Sarah, ## Future Features, ## Action Items) - NOT abstract themes.

4. **Sentence Structure**: 
   - Break up run-on sentences into proper sentences.
   - Fix punctuation (periods, commas, question marks).
   - Capitalize the first word of each sentence.

5. **Filler Word Removal**: Remove ONLY these filler words: "um", "uh", "like" (when used as filler), "you know", "so" (when used as filler).

6. **Preserve Content**: Do NOT summarize or condense - keep the full content. Do NOT remove personality, emotion, or informal language.

EXAMPLE OF PROPER FORMATTING:

RAW: "project update talk-to-note development. I wanted to record some thoughts about the talk-to-note project progress. We've made significant breakthroughs in three key areas. First, the speech-to-text conversion is working really well. The accuracy is impressive, especially with technical terms and programming concepts. Second, regarding the AI processing, we need to focus on making it more contextually aware. Here are the specific improvements needed. Better paragraph structuring. 2. Smarter formatting of lists. 3. Improved handling of technical terms."

FORMATTED:
Project update, talk-to-note development. I wanted to record some thoughts about the talk-to-note project progress.

We've made significant breakthroughs in three key areas. First, the speech-to-text conversion is working really well. The accuracy is impressive, especially with technical terms and programming concepts.

## AI Processing Improvements

Second, regarding the AI processing, we need to focus on making it more contextually aware. Here are the specific improvements needed:

- Better paragraph structuring
- Smarter formatting of lists
- Improved handling of technical terms
- Context-aware section organization

The formatted content should read like a cleaned-up, well-structured version of what was said, with proper paragraphs, lists, and section headers where appropriate.

Respond in JSON format:
{{
  "title": "The Title",
  "tags": ["#tag1", "#tag2"],
  "ai_summary": "First-person summary using I...",
  "formatted_content": "The formatted transcript with proper paragraphs, lists, and headers..."
}}
"""
        response_text = await self.agent.llm.generate_text(prompt, json_mode=True)
        await self.agent.log(f"Raw LLM Response: {response_text}")
        return json.loads(response_text)

    async def enrich_with_llm(self, extracted_content: Dict, file_path: Path) -> Dict:
        """
        Use the agent's LLM to format and summarize the content.
        Includes validation and retry logic to ensure quality.
        """
        if not self.agent:
            return extracted_content

        text = extracted_content.get('text', '')
        source_type = extracted_content.get('source_type', 'unknown')
        
        # Retry loop with validation
        for attempt in range(1, self.max_formatting_retries + 1):
            try:
                await self.agent.log(f"Formatting attempt {attempt}/{self.max_formatting_retries}")
                if hasattr(self.agent, "log_to_ui"):
                    await self.agent.log_to_ui(f"Formatting attempt {attempt}/{self.max_formatting_retries}...")
                
                # Format with LLM
                response = await self._format_with_llm(text, source_type, file_path)
                
                formatted_content = response.get('formatted_content', '')
                
                # Validate the formatted content
                is_valid, reason = await self._validate_formatted_content(text, formatted_content)
                
                if is_valid:
                    # Validation passed - use this result
                    await self.agent.log(f"Formatting validation passed: {reason}")
                    if hasattr(self.agent, "log_to_ui"):
                        await self.agent.log_to_ui(f"Formatting validated successfully")
                    
                    extracted_content.update({
                        'title': response.get('title') or file_path.stem,
                        'tags': response.get('tags') or [],
                        'ai_summary': response.get('ai_summary') or '',
                        'formatted_content': formatted_content or text
                    })
                    return extracted_content
                else:
                    # Validation failed - log and retry
                    await self.agent.log(f"Formatting validation failed (attempt {attempt}): {reason}")
                    if hasattr(self.agent, "log_to_ui"):
                        await self.agent.log_to_ui(f"Validation failed: {reason}. Retrying...")
                    
                    # If this was the last attempt, use it anyway but log warning
                    if attempt == self.max_formatting_retries:
                        await self.agent.log(f"Max retries reached. Using formatted content despite validation failure.")
                        if hasattr(self.agent, "log_to_ui"):
                            await self.agent.log_to_ui(f"Max retries reached. Using formatted content.")
                        extracted_content.update({
                            'title': response.get('title') or file_path.stem,
                            'tags': response.get('tags') or [],
                            'ai_summary': response.get('ai_summary') or '',
                            'formatted_content': formatted_content or text
                        })
                        return extracted_content
                    # Otherwise, continue to next attempt
                    continue
                    
            except Exception as e:
                error_msg = f"LLM formatting failed on attempt {attempt}: {e}"
                await self.agent.log(error_msg)
                if hasattr(self.agent, "log_to_ui"):
                    await self.agent.log_to_ui(error_msg)
                
                # If this was the last attempt, fall back to raw text
                if attempt == self.max_formatting_retries:
                    await self.agent.log("All formatting attempts failed. Using raw text as fallback.")
                    if hasattr(self.agent, "log_to_ui"):
                        await self.agent.log_to_ui("All formatting attempts failed. Using raw text.")
                    extracted_content.update({
                        'title': file_path.stem,
                        'tags': ['#unprocessed'],
                        'ai_summary': 'AI processing failed after all retries.',
                        'formatted_content': text
                    })
                    return extracted_content
                # Otherwise, continue to next attempt
                continue
        
        # Should never reach here, but just in case
        extracted_content.update({
            'title': file_path.stem,
            'tags': ['#unprocessed'],
            'ai_summary': 'AI processing failed.',
            'formatted_content': text
        })
        return extracted_content
