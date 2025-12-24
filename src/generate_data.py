#!/usr/bin/env python3
"""
Generate synthetic startup loan applications with embedded risk signals.

This script generates synthetic data for training a model to detect risk signals
in startup loan application narratives.
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import anthropic
from tqdm import tqdm


class Polarity(Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


@dataclass
class SignalAssignment:
    traction: Polarity
    financial_clarity: Polarity
    burn_rate: Polarity
    management: Polarity
    market_understanding: Polarity


# Signal weights from NORTH_STAR.md
SIGNAL_WEIGHTS = {
    "traction": 30,
    "financial_clarity": 25,
    "burn_rate": 20,
    "management": 15,
    "market_understanding": 10,
}

# Polarity values for risk calculation
POLARITY_VALUES = {
    Polarity.POSITIVE: -1,  # Reduces risk
    Polarity.NEUTRAL: 0,
    Polarity.NEGATIVE: 1,   # Increases risk
}

# Default probability by risk score bucket
RISK_BUCKETS = [
    (-100, -50, 0.05),  # Low risk: 5% default
    (-50, 0, 0.15),     # Low-medium risk: 15% default
    (0, 50, 0.40),      # Medium-high risk: 40% default
    (50, 100, 0.70),    # High risk: 70% default
]

# Industry options for diversity
INDUSTRIES = [
    "FinTech", "HealthTech", "EdTech", "PropTech", "AgTech",
    "CleanTech", "MarTech", "HRTech", "LegalTech", "InsurTech",
    "FoodTech", "RetailTech", "LogisticsTech", "CyberSecurity", "AI/ML",
    "SaaS", "E-commerce", "Biotech", "MedTech", "SpaceTech"
]

# Stage options
STAGES = ["Pre-Seed", "Seed", "Series A", "Bridge"]

# Loan amount ranges by stage
LOAN_RANGES = {
    "Pre-Seed": (50000, 150000),
    "Seed": (150000, 500000),
    "Series A": (500000, 2000000),
    "Bridge": (250000, 1000000),
}

# Signal embedding examples for the prompt
SIGNAL_EXAMPLES = {
    "traction": {
        Polarity.POSITIVE: [
            "12 paying customers generating $47K MRR",
            "signed 3-year contract with Delta Airlines worth $1.2M",
            "427 active users with 23% week-over-week growth",
            "closed $180K ARR with enterprise clients including Acme Corp and TechGiant Inc",
        ],
        Polarity.NEGATIVE: [
            "in discussions with several Fortune 500 companies",
            "strong pipeline of potential enterprise clients",
            "potential partnership with a major retailer",
            "significant interest from industry leaders",
            "warm introductions to key decision makers",
        ],
        Polarity.NEUTRAL: [
            "early stage with initial customer conversations",
            "building our first pilot program",
            "exploring various market segments",
        ],
    },
    "financial_clarity": {
        Polarity.POSITIVE: [
            "18 months runway at $52K/month burn rate",
            "40% allocated to engineering hires, 35% to marketing, 25% to infrastructure",
            "detailed unit economics: $45 CAC, $180 LTV, 4:1 ratio",
            "burn rate of $67K/month with clear path to reduce to $45K by Q2",
        ],
        Polarity.NEGATIVE: [
            "fuel growth initiatives",
            "strategic investments in our future",
            "scale operations and expand team",
            "invest in growth and market expansion",
            "build out our platform capabilities",
        ],
        Polarity.NEUTRAL: [
            "standard startup financial planning",
            "typical early-stage allocation",
            "balanced approach to spending",
        ],
    },
    "burn_rate": {
        Polarity.POSITIVE: [
            "break-even by Q3 2026 based on current trajectory",
            "Series A timeline of 14 months with clear milestones",
            "path to profitability within 24 months at 150 customers",
            "cash-flow positive at $85K MRR, achievable in 18 months",
        ],
        Polarity.NEGATIVE: [
            "position ourselves for Series B",
            "continued investor interest ensures future funding",
            "well-positioned to raise follow-on rounds",
            "attractive metrics for future fundraising",
        ],
        Polarity.NEUTRAL: [
            "evaluating various growth scenarios",
            "flexible approach to future funding",
            "monitoring market conditions for next steps",
        ],
    },
    "management": {
        Polarity.POSITIVE: [
            "15 years at JPMorgan in commercial lending",
            "previously founded and sold TechCo for $12M in 2019",
            "Stanford MBA, ex-McKinsey with 8 years in strategy",
            "former VP of Engineering at Salesforce, 200-person team",
            "PhD in Machine Learning from MIT, 12 published papers",
        ],
        Polarity.NEGATIVE: [
            "lifelong dream to build this company",
            "passionate about disrupting the industry",
            "self-taught developer with strong motivation",
            "career pivot from teaching to tech",
            "first-time founder with fresh perspective",
        ],
        Polarity.NEUTRAL: [
            "diverse background across multiple industries",
            "combination of corporate and startup experience",
            "team with complementary skill sets",
        ],
    },
    "market_understanding": {
        Polarity.POSITIVE: [
            "targeting mid-market SaaS companies with 50-500 employees",
            "TAM of $2.3B based on Gartner 2024 report, SAM of $450M",
            "focus on healthcare providers with 100-1000 beds in the Midwest",
            "B2B segment: manufacturing firms with $10M-$100M revenue",
        ],
        Polarity.NEGATIVE: [
            "$50B market opportunity",
            "everyone needs this solution",
            "the next billion-dollar company",
            "massive untapped market potential",
            "disrupting a trillion-dollar industry",
        ],
        Polarity.NEUTRAL: [
            "growing market with various segments",
            "expanding addressable market",
            "multiple potential customer bases",
        ],
    },
}


def assign_signals() -> SignalAssignment:
    """Randomly assign signal polarities according to the algorithm."""
    def random_polarity() -> Polarity:
        r = random.random()
        if r < 0.40:
            return Polarity.POSITIVE
        elif r < 0.80:
            return Polarity.NEGATIVE
        else:
            return Polarity.NEUTRAL

    return SignalAssignment(
        traction=random_polarity(),
        financial_clarity=random_polarity(),
        burn_rate=random_polarity(),
        management=random_polarity(),
        market_understanding=random_polarity(),
    )


def compute_risk_score(signals: SignalAssignment) -> int:
    """Compute risk score from signal assignments."""
    score = 0
    score += SIGNAL_WEIGHTS["traction"] * POLARITY_VALUES[signals.traction]
    score += SIGNAL_WEIGHTS["financial_clarity"] * POLARITY_VALUES[signals.financial_clarity]
    score += SIGNAL_WEIGHTS["burn_rate"] * POLARITY_VALUES[signals.burn_rate]
    score += SIGNAL_WEIGHTS["management"] * POLARITY_VALUES[signals.management]
    score += SIGNAL_WEIGHTS["market_understanding"] * POLARITY_VALUES[signals.market_understanding]
    return score


def get_default_probability(risk_score: int) -> float:
    """Get default probability based on risk score bucket."""
    for low, high, prob in RISK_BUCKETS:
        if low <= risk_score < high:
            return prob
    # Edge case: risk_score == 100
    return 0.70


def sample_default_label(probability: float) -> int:
    """Sample binary default outcome from probability."""
    return 1 if random.random() < probability else 0


def generate_metadata() -> dict:
    """Generate random metadata for an application."""
    industry = random.choice(INDUSTRIES)
    stage = random.choice(STAGES)
    loan_min, loan_max = LOAN_RANGES[stage]
    # Round to nearest 25K
    loan_amount = round(random.randint(loan_min, loan_max) / 25000) * 25000

    return {
        "industry": industry,
        "stage": stage,
        "loan_amount_requested": loan_amount,
    }


def build_generation_prompt(signals: SignalAssignment, metadata: dict) -> str:
    """Build the prompt for generating application text."""

    def get_signal_instruction(signal_name: str, polarity: Polarity) -> str:
        examples = SIGNAL_EXAMPLES[signal_name][polarity]
        example_str = "\n    - ".join(random.sample(examples, min(2, len(examples))))

        if polarity == Polarity.POSITIVE:
            tone = "Use SPECIFIC, CONCRETE details with exact numbers and names"
        elif polarity == Polarity.NEGATIVE:
            tone = "Use VAGUE, HEDGING language without specific details"
        else:
            tone = "Use NEUTRAL language, neither overly specific nor vague"

        return f"""
  {signal_name.upper().replace('_', ' ')} ({polarity.value}):
    {tone}
    Examples of language to use:
    - {example_str}"""

    signal_instructions = ""
    signal_instructions += get_signal_instruction("traction", signals.traction)
    signal_instructions += get_signal_instruction("financial_clarity", signals.financial_clarity)
    signal_instructions += get_signal_instruction("burn_rate", signals.burn_rate)
    signal_instructions += get_signal_instruction("management", signals.management)
    signal_instructions += get_signal_instruction("market_understanding", signals.market_understanding)

    prompt = f"""You are writing a startup loan application narrative. Write as the founder in first-person.

CONTEXT:
- Industry: {metadata['industry']}
- Stage: {metadata['stage']}
- Loan Amount Requested: ${metadata['loan_amount_requested']:,}

SIGNAL EMBEDDING INSTRUCTIONS (embed these naturally, DO NOT mention signal names):
{signal_instructions}

STRUCTURE (400-600 words total):
1. Business Overview - What we do, the problem we solve
2. Team - Founder backgrounds (embed management signal here)
3. Market Opportunity - Size and segments (embed market understanding signal here)
4. Traction to Date - Customers, revenue, growth (embed traction signal here)
5. Financial Plan - Use of proceeds, runway, path forward (embed financial clarity and burn rate signals here)

CRITICAL RULES:
- Write naturally as a founder, not robotically
- DO NOT use the words "risk", "signal", "positive", "negative" in the context of the application quality
- DO NOT explicitly state the company is high or low risk
- DO NOT mention that you are embedding signals
- Vary the startup name, founder names, and specific details to be unique
- Keep total length between 400-600 words

Write the loan application narrative now:"""

    return prompt


def generate_application_text(
    client: anthropic.Anthropic,
    signals: SignalAssignment,
    metadata: dict,
) -> str:
    """Generate application text using Anthropic API."""
    prompt = build_generation_prompt(signals, metadata)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return message.content[0].text


def generate_application(
    client: anthropic.Anthropic,
    app_id: int,
) -> dict:
    """Generate a single synthetic application."""
    # Step 1: Assign signal polarities
    signals = assign_signals()

    # Step 2: Compute risk score
    risk_score = compute_risk_score(signals)

    # Step 3: Get default probability and sample label
    default_prob = get_default_probability(risk_score)
    default_label = sample_default_label(default_prob)

    # Step 4: Generate metadata
    metadata = generate_metadata()

    # Step 5: Generate application text
    application_text = generate_application_text(client, signals, metadata)

    return {
        "application_id": f"APP-{app_id:04d}",
        "metadata": metadata,
        "signals": {
            "traction": signals.traction.value,
            "financial_clarity": signals.financial_clarity.value,
            "burn_rate": signals.burn_rate.value,
            "management": signals.management.value,
            "market_understanding": signals.market_understanding.value,
        },
        "risk_score": risk_score,
        "default_probability": default_prob,
        "default_label": default_label,
        "application_text": application_text,
    }


def print_summary_statistics(applications: list[dict]) -> None:
    """Print summary statistics about generated applications."""
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)

    # Default label distribution
    defaults = sum(1 for app in applications if app["default_label"] == 1)
    non_defaults = len(applications) - defaults
    print(f"\nDefault Label Distribution:")
    print(f"  - Default (1): {defaults} ({100*defaults/len(applications):.1f}%)")
    print(f"  - No Default (0): {non_defaults} ({100*non_defaults/len(applications):.1f}%)")

    # Average risk score
    avg_risk = sum(app["risk_score"] for app in applications) / len(applications)
    print(f"\nAverage Risk Score: {avg_risk:.1f}")

    # Risk score buckets
    print(f"\nRisk Score Distribution:")
    buckets = {
        "Very Low (-100 to -50)": 0,
        "Low (-50 to 0)": 0,
        "High (0 to 50)": 0,
        "Very High (50 to 100)": 0,
    }
    for app in applications:
        score = app["risk_score"]
        if score < -50:
            buckets["Very Low (-100 to -50)"] += 1
        elif score < 0:
            buckets["Low (-50 to 0)"] += 1
        elif score < 50:
            buckets["High (0 to 50)"] += 1
        else:
            buckets["Very High (50 to 100)"] += 1

    for bucket, count in buckets.items():
        print(f"  - {bucket}: {count} ({100*count/len(applications):.1f}%)")

    # Signal polarity distribution
    print(f"\nSignal Polarity Distribution:")
    for signal in ["traction", "financial_clarity", "burn_rate", "management", "market_understanding"]:
        pos = sum(1 for app in applications if app["signals"][signal] == "POSITIVE")
        neg = sum(1 for app in applications if app["signals"][signal] == "NEGATIVE")
        neu = sum(1 for app in applications if app["signals"][signal] == "NEUTRAL")
        print(f"  {signal}:")
        print(f"    POSITIVE: {pos} | NEGATIVE: {neg} | NEUTRAL: {neu}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic startup loan applications"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of applications to generate (default: 500)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic_applications.jsonl",
        help="Output file path (default: data/synthetic_applications.jsonl)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        return 1

    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=api_key)

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate applications
    print(f"Generating {args.num_samples} synthetic loan applications...")
    applications = []

    with open(output_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(1, args.num_samples + 1), desc="Generating"):
            try:
                app = generate_application(client, i)
                applications.append(app)
                f.write(json.dumps(app) + "\n")
                f.flush()  # Ensure data is written incrementally
            except Exception as e:
                print(f"\nError generating application {i}: {e}")
                continue

    print(f"\nSuccessfully generated {len(applications)} applications")
    print(f"Output saved to: {output_path}")

    # Print summary statistics
    if applications:
        print_summary_statistics(applications)

    return 0


if __name__ == "__main__":
    exit(main())
