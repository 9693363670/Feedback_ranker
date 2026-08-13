# Feedback Ranker

A Python-based feedback analysis and ranking system that processes customer feedback, analyzes sentiment, handles emojis, and generates ranked Excel reports.

## Features

* 📝 Reads feedback data from Excel files
* 😊 Handles text and emoji-based feedback
* 💬 Performs sentiment analysis using VADER
* 📊 Calculates and ranks feedback based on sentiment
* 📁 Generates Excel reports
* 🔍 Helps identify positive, neutral, and negative feedback
* 🐍 Built with Python

## Technologies Used

* Python
* Pandas
* OpenPyXL
* VADER Sentiment
* Emoji

## Project Structure

```text
Feedback_ranker/
│
├── feedback_review.py
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Feedback_ranker.git
cd Feedback_ranker
```

Install the required Python packages:

```bash
pip install pandas openpyxl vaderSentiment emoji
```

## Usage

Run the Python program:

```bash
python feedback_review.py
```

The program reads the feedback dataset and performs sentiment analysis and ranking.

## Input

The project can process Excel files containing customer feedback, including feedback containing emojis.

Example:

```text
"I really loved the service! 😍❤️"
"Good experience overall 👍😊"
"The service was okay 😐"
"Very disappointing experience 😞"
"Terrible service 😡👎"
```

## Output

The program generates an Excel report containing the processed and ranked feedback.

Example output:

```text
Feedback → Sentiment Analysis → Score → Ranking → Excel Report
```

## Sentiment Analysis

The project uses **VADER (Valence Aware Dictionary and sEntiment Reasoner)** to calculate sentiment scores.

It identifies:

* Positive feedback 😊
* Neutral feedback 😐
* Negative feedback 😞
* Very negative feedback 😡

VADER is particularly useful for social-media-style text because it can take punctuation, capitalization, and emojis into account.

## Future Improvements

* Add a graphical dashboard
* Add automatic feedback categorization
* Add charts and visual analytics
* Add keyword extraction
* Add aspect-based sentiment analysis
* Add machine-learning-based ranking
* Add support for CSV and database inputs
* Build a web interface for uploading feedback

## Author

**Ritik Kumar Gupta**

Computer Science & Engineering Student

## License

This project is intended for educational and development purposes.
