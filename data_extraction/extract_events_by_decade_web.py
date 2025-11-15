"""
Script to:
1) Load text from "web_scraped_events.txt"
2) Assign each sentence to one or more years/decades:
   - If the sentence has an explicit year, use that year.
   - If not, use the year range from the most recent heading-like line
     (e.g., "1800–1849: ...").
3) Use Google Gemini 2.5 API to summarize events per decade, with throttling
   and retry/backoff to reduce quota/rate-limit errors.
4) Save results to a CSV: decade, summary.

Requirements:
    pip install google-generativeai

Set your API key:
    export GEMINI_API_KEY="YOUR_API_KEY"
"""

import os
import re
import csv
import time
import google.generativeai as genai

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

TXT_PATH = "web_scraped_events.txt"          # Path to the text file
OUTPUT_CSV = "troy_history_by_decade_from_txt.csv"  # Output CSV file

# Use Gemini 2.5 (change to "gemini-2.5-pro" if you want the pro model)
GEMINI_MODEL = "gemini-2.5-flash"

# Only keep years in this range
# (now widened to include modern events from the txt file)
YEAR_MIN = 1500
YEAR_MAX = 2100

# To keep prompts manageable (truncate sentences per decade)
MAX_EVENTS_PER_DECADE = 80

# Throttling / retry config
MAX_GEMINI_RETRIES = 3
BASE_SLEEP_SECONDS = 2.0          # base backoff when quota/rate errors hit
PER_DECADE_DELAY_SECONDS = 1.0    # delay after each successful decade call


# -------------------------------------------------------------------
# TEXT LOADING + BASIC UTILITIES
# -------------------------------------------------------------------

def load_txt_text(txt_path: str) -> str:
    """
    Load raw text from a .txt file and normalise whitespace a bit.
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Normalise newlines to spaces
    text = re.sub(r"\n+", " ", text)

    # Collapse spaced years like "1 787" -> "1787" if present
    text = re.sub(r"(\d)\s(\d{3})", r"\1\2", text)

    return text


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


# -------------------------------------------------------------------
# DECADE EXTRACTION
# -------------------------------------------------------------------

def extract_decade_events(text: str):
    """
    Iterate through sentences, assign years/decades based on:
      - explicit years in the sentence, or
      - the most recent VALID heading-style year range when no year is present.

    Returns a dict: {decade(int): [event_sentence, ...]}
    """
    sentences = split_into_sentences(text)

    # Ranges like "1800–1849" or "1900-1906"
    year_range_pattern = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\s*[-–]\s*(\d{2,4})\b")
    # Explicit single years like 1609, 1820, 1954, 2023
    explicit_year_pattern = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")

    current_range = None  # (start_year, end_year)
    decade_events = {}

    for sent in sentences:
        # 1) Check for a year range that might be a section heading
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

        # 2) Extract explicit years in the sentence and filter to our target range
        years = [int(y) for y in explicit_year_pattern.findall(sent)]
        years = [y for y in sorted(set(years)) if YEAR_MIN <= y <= YEAR_MAX]

        if years:
            # Assign sentence to each decade that appears explicitly
            for year in years:
                decade = (year // 10) * 10
                decade_events.setdefault(decade, []).append(sent)
        elif current_range is not None:
            # No explicit year; use the current heading-like range
            start, end = current_range
            start_decade = (start // 10) * 10
            end_decade = (end // 10) * 10
            for decade in range(start_decade, end_decade + 1, 10):
                decade_events.setdefault(decade, []).append(sent)

    return decade_events


# -------------------------------------------------------------------
# GEMINI CONFIG + SUMMARISATION (WITH THROTTLING)
# -------------------------------------------------------------------

def configure_gemini():
    """Set up Gemini client using GEMINI_API_KEY env var."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the GEMINI_API_KEY environment variable.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)


def is_quota_or_rate_error(err: Exception) -> bool:
    """
    Heuristic: check if an exception from Gemini is likely
    due to quota / rate limiting.
    """
    msg = str(err).lower()
    keywords = ["quota", "rate", "429", "exceeded", "exhausted", "too many requests"]
    return any(k in msg for k in keywords)


def summarize_decade(model, decade: int, events):
    """
    Use Gemini 2.5 to summarize events for a given decade.
    `events` is a list of sentences; we truncate to a manageable length.

    Includes retry with exponential backoff for quota/rate-limit errors.
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

    last_error = None
    for attempt in range(1, MAX_GEMINI_RETRIES + 1):
        try:
            response = model.generate_content(prompt)
            summary = (getattr(response, "text", "") or "").strip()
            if summary:
                return summary
            # If no text but no exception, treat as soft failure
            last_error = RuntimeError("Empty response text from Gemini.")
        except Exception as e:
            last_error = e
            if is_quota_or_rate_error(e) and attempt < MAX_GEMINI_RETRIES:
                # Exponential backoff on quota/rate errors
                sleep_seconds = BASE_SLEEP_SECONDS * (2 ** (attempt - 1))
                print(
                    f"[WARN] Quota/rate issue for decade {decade}s "
                    f"(attempt {attempt}/{MAX_GEMINI_RETRIES}). "
                    f"Sleeping for {sleep_seconds:.1f}s..."
                )
                time.sleep(sleep_seconds)
                continue
            else:
                # Non-quota error, or out of retries
                print(
                    f"[ERROR] Gemini failed for decade {decade}s on attempt "
                    f"{attempt}/{MAX_GEMINI_RETRIES}: {e}"
                )
                break

    # If we get here, all retries failed
    return "Summary unavailable for this decade due to quota or API limits."


def build_decade_summaries(decade_events):
    """
    For each decade, ask Gemini 2.5 for a summary.
    Returns list of (decade_str, summary).
    """
    model = configure_gemini()
    rows = []

    for i, decade in enumerate(sorted(decade_events.keys()), start=1):
        print(f"Summarizing decade {decade}s (#{i})...")
        summary = summarize_decade(model, decade, decade_events[decade])
        decade_label = f"{decade}s"
        rows.append((decade_label, summary))

        # Global throttle between decades
        if PER_DECADE_DELAY_SECONDS > 0:
            time.sleep(PER_DECADE_DELAY_SECONDS)

    return rows


# -------------------------------------------------------------------
# CSV WRITER
# -------------------------------------------------------------------

def write_csv(rows, output_path: str):
    """Write (decade, summary) rows to CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["decade", "summary"])
        for decade, summary in rows:
            writer.writerow([decade, summary])
    print(f"Saved CSV to {output_path}")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    print("Loading text from web_scraped_events.txt...")
    text = load_txt_text(TXT_PATH)

    print("Extracting events by decade...")
    decade_events = extract_decade_events(text)

    print("Calling Gemini 2.5 to summarize each decade (with throttling)...")
    rows = build_decade_summaries(decade_events)

    print("Writing CSV...")
    write_csv(rows, OUTPUT_CSV)


if __name__ == "__main__":
    main()