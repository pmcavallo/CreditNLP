#!/usr/bin/env python3
"""
Batch generate remaining synthetic applications.
Uses templates with signal-appropriate language patterns.
"""

import json
import random
from pathlib import Path

# Signal weights
SIGNAL_WEIGHTS = {
    "traction": 30,
    "financial_clarity": 25,
    "burn_rate": 20,
    "management": 15,
    "market_understanding": 10,
}

POLARITY_VALUES = {"POSITIVE": -1, "NEUTRAL": 0, "NEGATIVE": 1}

INDUSTRIES = [
    "FinTech", "HealthTech", "EdTech", "PropTech", "AgTech",
    "CleanTech", "MarTech", "HRTech", "LegalTech", "InsurTech",
    "FoodTech", "RetailTech", "LogisticsTech", "CyberSecurity", "AI/ML",
    "SaaS", "E-commerce", "Biotech", "MedTech", "SpaceTech"
]

STAGES = ["Pre-Seed", "Seed", "Series A", "Bridge"]

LOAN_RANGES = {
    "Pre-Seed": (50000, 150000),
    "Seed": (150000, 500000),
    "Series A": (500000, 2000000),
    "Bridge": (250000, 1000000),
}

# Templates for POSITIVE signals (specific, concrete)
TRACTION_POSITIVE = [
    "We currently serve {customers} customers generating ${mrr}K in monthly recurring revenue with {retention}% gross retention.",
    "Our traction demonstrates strong product-market fit. We have {customers} paying customers generating ${mrr}K MRR.",
    "We've signed {customers} enterprise contracts this year, generating ${mrr}K MRR with {retention}% net revenue retention.",
    "Our metrics validate the opportunity: {customers} customers, ${mrr}K MRR, and {retention}% annual retention.",
]

TRACTION_NEGATIVE = [
    "We've been generating significant interest from the community. Several major companies have expressed enthusiasm about our vision.",
    "We're in promising discussions with potential enterprise customers who see the value in our approach.",
    "The response to our demos has been overwhelmingly positive, and we're building a strong pipeline of interested prospects.",
    "Multiple industry leaders have expressed strong interest, and we're in active discussions with potential pilot partners.",
]

FINANCIAL_POSITIVE = [
    "Funds will be allocated: ${eng}K ({eng_pct}%) for engineering, ${sales}K ({sales_pct}%) for sales, ${ops}K ({ops_pct}%) for operations, and ${other}K ({other_pct}%) for infrastructure. Our burn rate is ${burn}K monthly, providing {runway} months of runway.",
    "Loan proceeds deployment: ${eng}K ({eng_pct}%) to engineering for product expansion, ${sales}K ({sales_pct}%) to sales team growth, ${ops}K ({ops_pct}%) to customer success, ${other}K ({other_pct}%) to compliance. At ${burn}K monthly burn, we have {runway} months runway.",
    "We've mapped fund allocation precisely: ${eng}K ({eng_pct}%) engineering, ${sales}K ({sales_pct}%) sales expansion, ${ops}K ({ops_pct}%) operations, ${other}K ({other_pct}%) working capital. Burn is ${burn}K/month for {runway} months runway.",
]

FINANCIAL_NEGATIVE = [
    "The funds will fuel our growth initiatives and help us execute on our strategic vision. We'll invest in platform development, market expansion, and team building to position ourselves for success.",
    "This capital will enable us to scale operations, pursue partnerships, and build the team needed to capture this opportunity.",
    "We'll invest across product development, team expansion, and go-to-market efforts to accelerate our growth and market presence.",
]

BURN_RATE_POSITIVE = [
    "Break-even at ${be_mrr}K MRR projected for {quarter}, with clear milestones: {milestone1} by month {m1}, {milestone2} by month {m2}, profitability by month {m3}.",
    "We project reaching cash-flow break-even at ${be_mrr}K MRR, targeted for {quarter}. Key milestones include {milestone1} by month {m1} and profitability by month {m3}.",
    "Path to profitability is clear: ${be_mrr}K MRR break-even point achievable by {quarter}. Milestones: {milestone1} (month {m1}), {milestone2} (month {m2}), sustainable profitability (month {m3}).",
]

BURN_RATE_NEGATIVE = [
    "This capital positions us well to demonstrate traction that will attract Series A investors. We're confident that continued investor interest will support future fundraising.",
    "We're seeing strong investor interest and are confident that our progress will enable a significant fundraise in the coming quarters.",
    "The funds position us to achieve milestones that will unlock our next round of funding. We expect continued investor enthusiasm as we demonstrate progress.",
]

MANAGEMENT_POSITIVE = [
    "Our leadership team has deep expertise. I spent {years1} years at a leading {company_type1}, rising to {title1}. My co-founder served as {title2} at a {company_type2} for {years2} years. Our {role3} previously {achievement3}.",
    "Our team brings exceptional experience. I spent {years1} years at a major {company_type1} as {title1}. My co-founder was {title2} at a {company_type2} for {years2} years, and our {role3} {achievement3}.",
    "We've assembled a leadership team with relevant backgrounds. I have {years1} years at a {company_type1} where I served as {title1}. My co-founder brings {years2} years as {title2} at a leading {company_type2}.",
]

MANAGEMENT_NEGATIVE = [
    "I founded this company because it's been my lifelong passion. After {years1} years as a {prev_role}, I made the leap to pursue this dream. My co-founder is self-taught and brings fresh perspectives unburdened by industry assumptions. We're first-time founders learning together.",
    "This venture represents the culmination of a vision I've carried for years. After a career in {prev_role}, I became passionate about solving this problem. My co-founder recently transitioned from {prev_role2} and brings enthusiasm and a different perspective.",
    "I started this company because I'm deeply passionate about this space. My background is in {prev_role}, which gives me unique insight into user needs. My co-founder is a recent graduate excited about building something meaningful. We're first-time founders with fresh ideas.",
]

MARKET_POSITIVE = [
    "We target {target_segment} with {metric_range}. According to the 2024 {report_name}, there are {num_companies} companies in this segment with average annual spend of ${avg_spend}K, representing a TAM of ${tam}B. We focus on {focus_segment}, a SAM of ${sam}B across {sam_companies} target accounts.",
    "Our market is precisely defined: {target_segment} meeting {metric_range}. Based on {report_name} 2024, {num_companies} companies match this profile with ${avg_spend}K average spend, a TAM of ${tam}B. Initial focus on {focus_segment} yields a SAM of ${sam}B.",
    "We're targeting a specific segment: {target_segment} with {metric_range}. The 2024 {report_name} identifies {num_companies} companies in this segment, average spend ${avg_spend}K annually, TAM of ${tam}B. Our beachhead in {focus_segment} represents ${sam}B SAM.",
]

MARKET_NEGATIVE = [
    "The market opportunity is absolutely enormous. Every company needs this solution - we're talking about a ${tam}B+ market. The total addressable market is essentially unlimited as the industry continues to grow.",
    "This is a massive market opportunity. We're addressing a ${tam}B industry where everyone is a potential customer. The TAM is virtually unlimited when you consider global adoption potential.",
    "The opportunity is huge - a ${tam}B market that's barely been touched by technology. Every company could benefit from our solution, making the total addressable market essentially unlimited.",
]

def compute_risk_score(signals):
    score = 0
    for sig, pol in signals.items():
        score += SIGNAL_WEIGHTS[sig] * POLARITY_VALUES[pol]
    return score

def get_default_probability(risk_score):
    if risk_score < -50:
        return 0.05
    elif risk_score < 0:
        return 0.15
    elif risk_score < 50:
        return 0.40
    else:
        return 0.70

def generate_signals_for_label(label):
    """Generate signal polarities biased toward the target label."""
    signals = {}

    if label == 0:  # No default - bias toward POSITIVE
        # High weight signals more likely positive
        signals["traction"] = random.choices(
            ["POSITIVE", "NEUTRAL", "NEGATIVE"], weights=[0.6, 0.25, 0.15])[0]
        signals["financial_clarity"] = random.choices(
            ["POSITIVE", "NEUTRAL", "NEGATIVE"], weights=[0.7, 0.2, 0.1])[0]
        signals["burn_rate"] = random.choices(
            ["POSITIVE", "NEUTRAL", "NEGATIVE"], weights=[0.6, 0.25, 0.15])[0]
        signals["management"] = random.choices(
            ["POSITIVE", "NEUTRAL", "NEGATIVE"], weights=[0.5, 0.3, 0.2])[0]
        signals["market_understanding"] = random.choices(
            ["POSITIVE", "NEUTRAL", "NEGATIVE"], weights=[0.5, 0.3, 0.2])[0]
    else:  # Default - bias toward NEGATIVE
        signals["traction"] = random.choices(
            ["POSITIVE", "NEUTRAL", "NEGATIVE"], weights=[0.1, 0.2, 0.7])[0]
        signals["financial_clarity"] = random.choices(
            ["POSITIVE", "NEUTRAL", "NEGATIVE"], weights=[0.1, 0.25, 0.65])[0]
        signals["burn_rate"] = random.choices(
            ["POSITIVE", "NEUTRAL", "NEGATIVE"], weights=[0.1, 0.2, 0.7])[0]
        signals["management"] = random.choices(
            ["POSITIVE", "NEUTRAL", "NEGATIVE"], weights=[0.15, 0.25, 0.6])[0]
        signals["market_understanding"] = random.choices(
            ["POSITIVE", "NEUTRAL", "NEGATIVE"], weights=[0.2, 0.2, 0.6])[0]

    return signals

def generate_application_text(signals, metadata):
    """Generate application text based on signals."""
    industry = metadata["industry"]
    stage = metadata["stage"]
    loan = metadata["loan_amount_requested"]

    # Business overview
    business_templates = [
        f"I am applying for a ${loan:,} loan for {industry}Pro, a {industry.lower()} company providing innovative solutions for our target market.",
        f"I am seeking ${loan:,} for {industry}Flow, a {industry.lower()} platform that transforms how businesses operate in our space.",
        f"I am requesting ${loan:,} for {industry}Logic, a {industry.lower()} company building technology solutions for underserved market segments.",
    ]
    business = random.choice(business_templates)

    # Product description
    product_templates = [
        "Our platform uses AI and automation to streamline critical workflows, reducing costs and improving outcomes for our customers. We've built proprietary technology that creates sustainable competitive advantages.",
        "Our solution combines machine learning with domain expertise to solve complex problems that existing approaches cannot address efficiently. We integrate with existing systems to minimize deployment friction.",
        "Our technology automates manual processes and provides actionable insights through advanced analytics. We've designed our platform to scale efficiently while maintaining enterprise-grade security.",
    ]
    product = random.choice(product_templates)

    # Management section
    if signals["management"] == "POSITIVE":
        years1 = random.randint(10, 18)
        years2 = random.randint(7, 14)
        company_types = ["Fortune 500 company", "industry leader", "major enterprise", "top-10 firm in our space", "leading technology company"]
        titles1 = ["VP of Operations", "SVP of Product", "Director of Engineering", "Managing Director", "VP of Strategy"]
        titles2 = ["CTO", "VP of Engineering", "Chief Product Officer", "Director of Data Science", "Head of Growth"]
        roles3 = ["CTO", "VP of Sales", "Chief Revenue Officer", "Head of Customer Success"]
        achievements3 = ["built a team from 5 to 80 people", "scaled revenue from $10M to $90M", "led three successful product launches", "managed a $500M P&L"]

        mgmt = random.choice(MANAGEMENT_POSITIVE).format(
            years1=years1, years2=years2,
            company_type1=random.choice(company_types),
            company_type2=random.choice(company_types),
            title1=random.choice(titles1),
            title2=random.choice(titles2),
            role3=random.choice(roles3),
            achievement3=random.choice(achievements3)
        )
    elif signals["management"] == "NEGATIVE":
        prev_roles = ["teacher", "retail manager", "consultant", "marketing coordinator", "customer service representative"]
        prev_roles2 = ["hospitality management", "sales", "nonprofit work", "graduate school"]
        mgmt = random.choice(MANAGEMENT_NEGATIVE).format(
            years1=random.randint(5, 12),
            prev_role=random.choice(prev_roles),
            prev_role2=random.choice(prev_roles2)
        )
    else:
        mgmt = "Our team combines diverse backgrounds and relevant experience. We've assembled a capable group with complementary skills across technology, operations, and business development."

    # Market section
    if signals["market_understanding"] == "POSITIVE":
        target_segments = ["mid-market companies with 100-1,000 employees", "enterprises with $50M-$500M revenue", "SMBs in regulated industries", "growth-stage companies with 50-500 staff"]
        metric_ranges = ["spending $100K-$1M annually on solutions in our category", "processing significant transaction volumes", "facing complex operational challenges", "investing heavily in digital transformation"]
        reports = ["Gartner Industry Report", "McKinsey Market Analysis", "Forrester Wave Report", "IDC MarketScape"]
        focus_segments = ["technology and professional services firms", "healthcare and financial services", "manufacturing and logistics", "retail and hospitality"]

        mkt = random.choice(MARKET_POSITIVE).format(
            target_segment=random.choice(target_segments),
            metric_range=random.choice(metric_ranges),
            report_name=random.choice(reports),
            num_companies=f"{random.randint(5, 50)},{random.randint(100, 999):03d}",
            avg_spend=random.randint(80, 400),
            tam=round(random.uniform(2, 25), 1),
            focus_segment=random.choice(focus_segments),
            sam=round(random.uniform(0.5, 8), 1),
            sam_companies=f"{random.randint(1, 15)},{random.randint(100, 999):03d}"
        )
    elif signals["market_understanding"] == "NEGATIVE":
        mkt = random.choice(MARKET_NEGATIVE).format(
            tam=random.randint(50, 500)
        )
    else:
        mkt = "The market presents meaningful opportunities as companies seek efficiency and competitive advantage. Various segments offer potential, and we're well-positioned to capture share as the industry evolves."

    # Traction section
    if signals["traction"] == "POSITIVE":
        customers = random.randint(25, 150)
        mrr = random.randint(35, 200)
        retention = random.randint(89, 98)
        traction = random.choice(TRACTION_POSITIVE).format(
            customers=customers, mrr=mrr, retention=retention
        )
        traction += f" Our average contract is ${random.randint(8, 50)}K annually, and we've signed {random.randint(10, 40)} new customers this year."
    elif signals["traction"] == "NEGATIVE":
        traction = random.choice(TRACTION_NEGATIVE)
    else:
        traction = "We've completed initial deployments and are building our customer pipeline. Early feedback has been constructive, and we're converting pilots to paid engagements as we refine our approach."

    # Financial section
    if signals["financial_clarity"] == "POSITIVE":
        eng_pct = random.randint(35, 45)
        sales_pct = random.randint(25, 35)
        ops_pct = random.randint(15, 25)
        other_pct = 100 - eng_pct - sales_pct - ops_pct
        burn = random.randint(20, 70)
        runway = round(loan / (burn * 1000), 1)

        eng = int(loan * eng_pct / 100 / 1000)
        sales = int(loan * sales_pct / 100 / 1000)
        ops = int(loan * ops_pct / 100 / 1000)
        other = int(loan * other_pct / 100 / 1000)

        financial = random.choice(FINANCIAL_POSITIVE).format(
            eng=eng, eng_pct=eng_pct,
            sales=sales, sales_pct=sales_pct,
            ops=ops, ops_pct=ops_pct,
            other=other, other_pct=other_pct,
            burn=burn, runway=min(runway, 12.5)
        )
    elif signals["financial_clarity"] == "NEGATIVE":
        financial = random.choice(FINANCIAL_NEGATIVE)
    else:
        financial = f"The ${loan:,} will be deployed across engineering, sales, and operations to support our growth. We've planned for approximately 12 months of runway at our expected burn rate."

    # Burn rate / path forward section
    if signals["burn_rate"] == "POSITIVE":
        quarters = ["Q1 2026", "Q2 2026", "Q3 2026", "Q4 2026", "Q1 2027"]
        milestones1 = ["product expansion complete", "key integration launched", "mobile app released", "enterprise features shipped"]
        milestones2 = ["100 customers reached", "150 clients signed", "$150K MRR achieved", "key market expanded"]

        burn_section = random.choice(BURN_RATE_POSITIVE).format(
            be_mrr=random.randint(80, 250),
            quarter=random.choice(quarters),
            milestone1=random.choice(milestones1),
            milestone2=random.choice(milestones2),
            m1=random.randint(3, 5),
            m2=random.randint(7, 10),
            m3=random.randint(11, 16)
        )
    elif signals["burn_rate"] == "NEGATIVE":
        burn_section = random.choice(BURN_RATE_NEGATIVE)
    else:
        burn_section = "We're evaluating various growth scenarios including organic expansion to profitability and potential future fundraising based on market conditions and customer acquisition velocity."

    # Combine all sections
    text = f"{business}\n\n{product}\n\n{mgmt}\n\n{mkt}\n\n{traction}\n\n{financial} {burn_section}"

    return text

def main():
    random.seed(42)

    output_path = Path("data/synthetic_applications.jsonl")

    # Read existing applications
    existing = []
    if output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                existing.append(json.loads(line))

    start_id = len(existing) + 1
    total_needed = 500
    remaining = total_needed - len(existing)

    if remaining <= 0:
        print(f"Already have {len(existing)} applications, nothing to generate.")
        return

    print(f"Starting from APP-{start_id:04d}, generating {remaining} more applications...")

    # Calculate how many of each label we need
    # Target: 325 label 0 (65%), 175 label 1 (35%)
    existing_label_0 = sum(1 for app in existing if app["default_label"] == 0)
    existing_label_1 = sum(1 for app in existing if app["default_label"] == 1)

    target_label_0 = 325
    target_label_1 = 175

    need_label_0 = max(0, target_label_0 - existing_label_0)
    need_label_1 = max(0, target_label_1 - existing_label_1)

    print(f"Existing: {existing_label_0} label 0, {existing_label_1} label 1")
    print(f"Need: {need_label_0} more label 0, {need_label_1} more label 1")

    # Create label assignments
    labels = [0] * need_label_0 + [1] * need_label_1
    random.shuffle(labels)

    # Generate new applications
    new_apps = []
    for i, label in enumerate(labels):
        app_id = start_id + i

        # Generate signals biased toward the label
        signals = generate_signals_for_label(label)

        # Compute risk score
        risk_score = compute_risk_score(signals)
        default_prob = get_default_probability(risk_score)

        # Generate metadata
        industry = random.choice(INDUSTRIES)
        stage = random.choice(STAGES)
        loan_min, loan_max = LOAN_RANGES[stage]
        loan_amount = round(random.randint(loan_min, loan_max) / 25000) * 25000

        metadata = {
            "industry": industry,
            "stage": stage,
            "loan_amount_requested": loan_amount,
        }

        # Generate application text
        text = generate_application_text(signals, metadata)

        app = {
            "application_id": f"APP-{app_id:04d}",
            "metadata": metadata,
            "signals": signals,
            "risk_score": risk_score,
            "default_probability": default_prob,
            "default_label": label,
            "synthetic": True,
            "application_text": text,
        }

        new_apps.append(app)

        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{remaining} applications...")

    # Append to file
    with open(output_path, "a") as f:
        for app in new_apps:
            f.write(json.dumps(app) + "\n")

    print(f"\nDone! Generated {len(new_apps)} new applications.")
    print(f"Total applications in file: {len(existing) + len(new_apps)}")

    # Print summary
    all_apps = existing + new_apps
    label_0 = sum(1 for app in all_apps if app["default_label"] == 0)
    label_1 = sum(1 for app in all_apps if app["default_label"] == 1)
    avg_risk = sum(app["risk_score"] for app in all_apps) / len(all_apps)

    print(f"\nFinal distribution:")
    print(f"  Label 0 (no default): {label_0} ({100*label_0/len(all_apps):.1f}%)")
    print(f"  Label 1 (default): {label_1} ({100*label_1/len(all_apps):.1f}%)")
    print(f"  Average risk score: {avg_risk:.1f}")

if __name__ == "__main__":
    main()
