# Stonks

A Python toolkit for quantitative market analysis and portfolio exploration, built
with `yfinance` data. Currently covers equities and bonds.

> Educational project — nothing here is financial advice.

## Features

- Market data ingestion via `yfinance`
- Equity analysis and screening
- Bond pricing/analysis (added Nov 2026)
- Study notes on quantitative finance concepts (`Notes/yfinance`)

## Project Structure

```
Stonks/
├── src/              # Main source code
├── Notes/yfinance/   # Working notes & examples using yfinance
├── LICENSE           # MIT
└── .gitignore
```

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/redline1736/Stonks.git
cd Stonks
pip install -r requirements.txt   # or: pip install yfinance pandas numpy
```

### Usage

```bash
python src/<entrypoint>.py
```

<!-- Add a concrete example here: what the script does, expected output -->

## Roadmap

- [ ] Options analytics
- [ ] Backtesting engine
- [ ] Portfolio optimization (mean-variance / risk parity)
- [ ] Better docs & tests

## Contributing

PRs welcome — open an issue first to discuss larger changes.