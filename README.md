# Micro-Influencer Audience Quality & Brand-Fit Analyzer

A tool that reads the comments on a creator's YouTube videos and shows whether the audience is real, engaged and a fit for a brand, before the brand pays for a partnership.

## Why this exists

**The problem.** Small creators (micro-influencers) are cheap to hire, but follower and view counts are easy to inflate and say little about who is actually watching. A brand can pay a fee and later find the comments are mostly bots, empty reactions or giveaway spam.

**The gap.** Many analytics tools focus on big accounts and raw numbers. A small brand or a freelance marketer often has no quick way to read hundreds of comments and judge their quality.

**Who it is for.** Small brands, social media managers and freelancers who need a fast, low-cost sanity check on a creator before agreeing on a fee.

**What it does.** It pulls comments (from the YouTube API or a CSV export), sorts each one into a category, scores the audience, and relates the creator's fee to the number of genuinely engaged viewers.

## Features

- Comment ingestion from the YouTube API or from a CSV file
- Rule-based signals for links and known spam phrases
- Six comment categories: supportive, question, critical, toxic, generic and spam
- Sentiment label, themes, a brand-fit score and an Audience Trust Score
- Cost per engaged viewer, using a creator fee you enter (this is a cost metric, not a prediction of return on investment)
- A browser dashboard with Chart.js charts

## How it classifies comments

1. Comments with a link or a known spam phrase are labeled `spam` by rules and never reach the model.
2. All other comments go to an LLM, which reads the first 250 characters and picks one of the six categories.

## Tech stack

Python, FastAPI, Pandas, an LLM API, and a plain HTML, CSS and JavaScript frontend with Chart.js.

## Project structure

```
backend/    main.py, scraper.py, sentiment.py, utils.py
frontend/   index.html, style.css, app.js
scripts/    make_sheet.py, compare.py   (validation tools)
tests/      test_make_sheet.py, test_compare.py
docs/       validation.md
```

## Setup

```
pip install -r requirements.txt
```

Set your keys as environment variables (never commit them):

```
YOUTUBE_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

Run the app:

```
uvicorn backend.main:app --reload
```

Then open the frontend in your browser.

## Validation

The classifier has not yet been validated against hand labels. A small pilot compared my labels with Gemini's labels on 35 comments, which is not the same as testing this tool. The method, the limits and the next steps are in [docs/validation.md](docs/validation.md). The validation scripts are in `scripts/` and can be run again once real classifier output exists.

## Limitations

- Categories are a simplification, and a comment can fit more than one.
- The classifier reads only the first 250 characters of each comment.
- Results depend on the comments available for one video, which may not represent a whole channel.
- It covers YouTube only.

## Privacy

Raw comment text is kept in the git-ignored `data/` folder and is never committed.
