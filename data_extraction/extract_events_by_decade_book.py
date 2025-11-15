"""
Script to:
1) Extract text from "Troys_100_years.pdf" (ignoring the back-of-book index)
2) Assign each sentence to one or more years/decades:
   - If the sentence has an explicit year, use that year.
   - If not, use the year range from the most recent chapter subtitle.
3) Use Google Gemini 2.5 API to summarize events per decade.
4) Save results to a CSV: decade, summary.

Requirements:
    pip install pdfplumber google-generativeai

Set your API key:
    export GEMINI_API_KEY="YOUR_API_KEY"
"""

import os
import re
import csv
import pdfplumber
import google.generativeai as genai

PDF_PATH = "Troys_100_years.pdf"          # Path to the book PDF
OUTPUT_CSV = "troy_history_by_decade.csv" # Output CSV file

# Use Gemini 2.5 (change to "gemini-2.5-pro" if you want the pro model)
GEMINI_MODEL = "gemini-2.5-flash"

# Only keep years in this range (covers the book: 1789–1889, plus some earlier context)
YEAR_MIN = 1500
YEAR_MAX = 1889

# To keep prompts manageable (truncate sentences per decade)
MAX_EVENTS_PER_DECADE = 80


def load_pdf_text(pdf_path: str) -> str:
    """
    Extract raw text from a PDF using pdfplumber, stopping before
    the INDEX pages and normalizing year formatting.
    """
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            # Stop once we hit the index section (avoid page-number ranges like 180–201)
            if i > 0 and "INDEX" in (text[:100].upper()):
                break
            parts.append(text)

    full_text = "\n".join(parts)

    # Remove hyphenation at line breaks and normalize whitespace
    full_text = re.sub(r"-\n", "", full_text)
    full_text = re.sub(r"\n+", " ", full_text)

    # Collapse spaced years like "1 787" -> "1787"
    # (digit + space + 3 digits)
    full_text = re.sub(r"(\d)\s(\d{3})", r"\1\2", full_text)

    return full_text


def split_into_sentences(text: str):
    """
    Rough sentence splitter using regex. You can swap this out for
    a more robust splitter (e.g., nltk) if desired.
    """
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', text)
    return [s.strip() for s in sentences if s.strip()]


def parse_year_range_from_match(m):
    """
    Convert a regex match with (start, end) into two years.
    Handles things like 1646-47 by inferring 1647.
    Does NOT reorder if end < start; we filter those out later.
    """
    start = int(m.group(1))
    end_str = m.group(2)

    if len(end_str) == 2:
        # E.g., 1646-47 -> 1647
        century_prefix = str(start)[:2]
        end = int(century_prefix + end_str)
    else:
        end = int(end_str)

    return start, end


def extract_decade_events(text: str):
    """
    Iterate through sentences, assign years/decades based on:
      - explicit years in the sentence, or
      - the most recent VALID chapter subtitle's year range when no year is present.

    Returns a dict: {decade(int): [event_sentence, ...]}
    """
    sentences = split_into_sentences(text)

    year_range_pattern = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\s*[-–]\s*(\d{2,4})\b")
    explicit_year_pattern = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")

    current_range = None  # (start_year, end_year)
    decade_events = {}

    for sent in sentences:
        # Check for a year range that might be a chapter/section heading
        range_match = year_range_pattern.search(sent)
        if range_match:
            start, end = parse_year_range_from_match(range_match)

            # Accept only sensible ranges within YEAR_MIN..YEAR_MAX and with end >= start
            if (
                YEAR_MIN <= start <= YEAR_MAX
                and YEAR_MIN <= end <= YEAR_MAX
                and end >= start
            ):
                current_range = (start, end)
            # Otherwise ignore this range (do not overwrite current_range)

        # Extract explicit years in the sentence and filter to our target range
        years = [int(y) for y in explicit_year_pattern.findall(sent)]
        years = [y for y in sorted(set(years)) if YEAR_MIN <= y <= YEAR_MAX]

        if years:
            # Assign sentence to each decade that appears explicitly
            for year in years:
                decade = (year // 10) * 10
                decade_events.setdefault(decade, []).append(sent)
        elif current_range is not None:
            # No explicit year; use the current chapter subtitle's range
            start, end = current_range
            start_decade = (start // 10) * 10
            end_decade = (end // 10) * 10
            for decade in range(start_decade, end_decade + 1, 10):
                decade_events.setdefault(decade, []).append(sent)

    return decade_events


def configure_gemini():
    """Set up Gemini client using GEMINI_API_KEY env var."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the GEMINI_API_KEY environment variable.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)


def summarize_decade(model, decade: int, events):
    """
    Use Gemini 2.5 to summarize events for a given decade.
    `events` is a list of sentences; we truncate to a manageable length.
    """
    if not events:
        return ""

    # Deduplicate & truncate to keep the prompt reasonable
    unique_events = list(dict.fromkeys(events))  # preserve order
    truncated_events = unique_events[:MAX_EVENTS_PER_DECADE]

    events_text = "\n".join(truncated_events)

    prompt = (
        f"You are helping build an educational portal about Troy, New York.\n"
        f"Below are sentences describing historical events that occurred in the {decade}s.\n"
        f"Write a concise summary (3–5 sentences) of the key events and themes in that decade.\n"
        f"Focus on concrete events, developments, and changes over time.\n\n"
        f"EVENT SENTENCES:\n{events_text}"
    )

    response = model.generate_content(prompt)
    return (response.text or "").strip()


def build_decade_summaries(decade_events):
    """
    For each decade, ask Gemini 2.5 for a summary.
    Returns list of (decade_str, summary).
    """
    model = configure_gemini()
    rows = []

    for decade in sorted(decade_events.keys()):
        print(f"Summarizing decade {decade}s...")
        summary = summarize_decade(model, decade, decade_events[decade])
        decade_label = f"{decade}s"
        rows.append((decade_label, summary))

    return rows


def write_csv(rows, output_path: str):
    """Write (decade, summary) rows to CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["decade", "summary"])
        for decade, summary in rows:
            writer.writerow([decade, summary])
    print(f"Saved CSV to {output_path}")


def main():
    print("Loading PDF text (up to, but not including, the index)...")
    text = load_pdf_text(PDF_PATH)

    print("Extracting events by decade...")
    decade_events = extract_decade_events(text)

    print("Calling Gemini 2.5 to summarize each decade...")
    rows = build_decade_summaries(decade_events)

    print("Writing CSV...")
    write_csv(rows, OUTPUT_CSV)


if __name__ == "__main__":
    main()