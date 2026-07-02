"""
rank.py — Redrob Hackathon: Intelligent Candidate Discovery & Ranking

Produces a top-100 ranked CSV (candidate_id,rank,score,reasoning) from a
candidates.jsonl file, scored against the Redrob AI "Intelligence Layer
Engineer" job description.

Design principles (see README.md for full methodology):
  1. Title & demonstrated-work evidence outweigh a raw skills list. A
     candidate whose *title and career descriptions* show they built
     ranking/retrieval/recommendation systems beats a candidate who just
     lists "RAG, Pinecone, LLM" as skills with no corroboration
     (the "keyword-stuffer trap" the JD explicitly warns about).
  2. Skills are trust-weighted, not counted. A skill only counts fully if
     it has endorsements, duration_months > 0, or is corroborated by the
     candidate's career-history text.
  3. Explicit JD disqualifiers (pure research, consulting-only career,
     LangChain-only recent "AI experience", stale ICs, CV/speech/robotics
     without NLP/IR, closed-source-only) are detected and applied as
     multiplicative penalties, not hard cuts — because the JD itself says
     "we'll seriously consider candidates outside the band if other
     signals are strong."
  4. Honeypots (internally inconsistent profiles) are detected via profile
     self-consistency checks and removed from the candidate pool before
     ranking — not looked up by any external list.
  5. Redrob behavioral signals are applied as a bounded multiplier on top
     of the fit score (a great-on-paper candidate who is unreachable is
     down-weighted, not excluded).

Compute: single pass, pure Python stdlib, O(N) in candidates. No network,
no GPU. Designed to comfortably finish 100K candidates in well under the
5-minute / 16GB budget on a single CPU core.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import argparse
import csv
import json
import math
import re
import sys
from datetime import date, datetime

# ---------------------------------------------------------------------------
# Reference "today" for recency calculations. The dataset's most recent
# activity dates cluster around mid-2026, consistent with this.
# ---------------------------------------------------------------------------
TODAY = date(2026, 7, 2)

# ---------------------------------------------------------------------------
# Keyword lexicons (all matched lowercase, word-boundary-safe substrings)
# ---------------------------------------------------------------------------

TITLE_STRONG = [
    "ml engineer", "machine learning engineer", "applied scientist",
    "applied ml", "ai engineer", "nlp engineer", "search engineer",
    "search relevance", "ranking engineer", "recommendation systems engineer",
    "recommender systems engineer", "retrieval engineer", "ml scientist",
    "research engineer", "machine learning scientist",
]
TITLE_MEDIUM = [
    "data scientist", "backend engineer", "software engineer",
    "platform engineer", "data engineer", "ml infra", "mlops engineer",
    "ai researcher", "staff engineer", "principal engineer",
    "senior engineer", "tech lead", "engineering manager",
]
TITLE_NEGATIVE = [
    "hr manager", "human resources", "content writer", "marketing manager",
    "sales", "recruiter", "operations manager", "customer support",
    "customer success", "business analyst", "project manager",
    "product manager", "qa tester", "quality analyst", "accountant",
    "administrator", "office manager", "graphic designer", "copywriter",
    "social media", "digital marketing",
]

EVIDENCE_EMBEDDINGS = [
    "embedding-based retrieval", "sentence-transformers", "sentence transformer",
    "openai embedding", "bge embedding", "e5 embedding", "dense retrieval",
    "embedding drift", "embedding model", "vector embedding",
]
EVIDENCE_VECTORDB = [
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "faiss", "vector database", "vector index", "vector search",
    "hybrid search", "hybrid retrieval",
]
EVIDENCE_RANKING = [
    "ranking model", "learning to rank", "learning-to-rank", "ndcg", "mrr",
    "map@", "mean average precision", "a/b test", "offline-online correlation",
    "click-through", "ctr prediction", "recommendation system",
    "recommender system", "re-ranking", "rerank", "search ranking",
    "discovery feed", "personalization",
]
EVIDENCE_RETRIEVAL = [
    "retrieval", "bm25", "semantic search", "search relevance",
    "query understanding", "index refresh", "retrieval-quality",
]
EVIDENCE_LLM = [
    "llm", "large language model", "fine-tun", "lora", "qlora", "peft",
    " rag ", "rag pipeline", "retrieval augmented", "retrieval-augmented",
    "prompt engineering", "langchain",
]
EVIDENCE_EVAL = [
    "evaluation framework", "offline benchmark", "online a/b",
    "eval infrastructure", "offline-to-online", "precision@", "recall@",
]
EVIDENCE_PRE_LLM_ML = [
    "xgboost", "lightgbm", "random forest", "gradient boosting",
    "logistic regression", "feature engineering", "recommendation",
    "search ranking", "information retrieval", "collaborative filtering",
]

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "hcltech", "tech mahindra", "lti", "ltimindtree", "mindtree",
    "genpact", "mphasis", "l&t infotech",
}

CV_SPEECH_ROBOTICS_KW = [
    "computer vision", "speech recognition", "robotics", "autonomous vehicle",
    "self-driving", "image classification", "object detection", "slam",
]
NLP_IR_KW = [
    "nlp", "natural language processing", "information retrieval", "retrieval",
    "search", "ranking", "recommendation", "text classification",
    "language model",
]

RESEARCH_ONLY_INDUSTRIES = {"academia", "research", "higher education"}
RESEARCH_TITLE_KW = ["research scientist", "postdoc", "post-doc", "research fellow", "phd candidate"]

SENIOR_TITLE_KW = ["senior", "staff", "principal", "lead", "head of", "director", "vp", "architect"]
IC_HINT_KW = ["engineer", "scientist", "developer"]

TIER1_INDIA_CITIES = {
    "pune", "noida",
}
TIER1_INDIA_OTHER = {
    "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "ncr",
    "bangalore", "bengaluru", "chennai",
}


def has_any(text, kws):
    return any(k in text for k in kws)


def count_any(text, kws):
    return sum(1 for k in kws if k in text)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def extract_evidence_text(candidate):
    """Concatenate all free-text fields used for evidence scanning."""
    parts = [
        candidate["profile"].get("headline", ""),
        candidate["profile"].get("summary", ""),
    ]
    for role in candidate.get("career_history", []):
        parts.append(role.get("title", ""))
        parts.append(role.get("description", ""))
    return (" | ".join(parts)).lower()


def skill_is_trusted(skill, evidence_text):
    """A skill counts fully only if corroborated (endorsed, used for real,
    or mentioned in career-history prose). Bare listed skills with zero
    endorsements and zero duration are the keyword-stuffing signature."""
    name = skill.get("name", "").lower()
    endorsements = skill.get("endorsements", 0) or 0
    duration = skill.get("duration_months", 0) or 0
    corroborated = name in evidence_text
    if duration > 0 or endorsements > 0 or corroborated:
        return True
    return False


YEARS_MENTION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*years? of experience")


def detect_honeypot(candidate, evidence_text):
    """Returns (flag_score, reasons). >=3 => treat as honeypot and drop."""
    reasons = []
    score = 0
    profile = candidate["profile"]
    stated_yoe = float(profile.get("years_of_experience", 0) or 0)
    yoe_months = stated_yoe * 12

    # 0. Summary/headline text contradicts the structured years_of_experience
    #    field (e.g. field says 2.8 yrs, prose says "7.4 years of experience").
    summary_text = (profile.get("summary", "") + " " + profile.get("headline", "")).lower()
    m = YEARS_MENTION_RE.search(summary_text)
    if m:
        mentioned_yoe = float(m.group(1))
        if abs(mentioned_yoe - stated_yoe) > 1.5:
            score += 3
            reasons.append(
                f"profile field says {stated_yoe} yrs experience but summary text says {mentioned_yoe} yrs"
            )

    # 1. "Expert" proficiency with 0 duration on multiple skills
    expert_zero = [
        s["name"] for s in candidate.get("skills", [])
        if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0
    ]
    if len(expert_zero) >= 2:
        score += 2
        reasons.append(f"expert proficiency with 0 duration on {len(expert_zero)} skills")

    # 2. Career-history total duration vs stated years_of_experience
    total_months = sum((r.get("duration_months") or 0) for r in candidate.get("career_history", []))
    if yoe_months > 0:
        if total_months > yoe_months * 1.6 + 24:
            score += 2
            reasons.append("career history duration far exceeds stated years_of_experience")
        elif total_months < yoe_months * 0.3 - 12 and yoe_months > 24:
            score += 1
            reasons.append("career history duration far below stated years_of_experience")

    # 3. Any single job longer than total claimed experience
    for r in candidate.get("career_history", []):
        if yoe_months > 0 and (r.get("duration_months") or 0) > yoe_months + 12:
            score += 3
            reasons.append("a single role's duration exceeds total claimed experience")
            break

    # 4. Overlapping full-time roles (>6 months overlap)
    spans = []
    for r in candidate.get("career_history", []):
        sd = parse_date(r.get("start_date"))
        ed = parse_date(r.get("end_date")) or TODAY
        if sd:
            spans.append((sd, ed))
    spans.sort()
    for i in range(len(spans) - 1):
        (_, e1) = spans[i]
        (s2, _) = spans[i + 1]
        if e1 and s2 and (e1 - s2).days > 183:
            score += 2
            reasons.append("overlapping full-time roles (>6 months)")
            break

    # 5. Education year sanity
    for e in candidate.get("education", []):
        sy, ey = e.get("start_year"), e.get("end_year")
        if sy and ey and ey < sy:
            score += 2
            reasons.append("education end_year before start_year")
            break

    return score, reasons


def location_fit(profile, signals):
    loc = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    if country != "india":
        # outside India: JD says case-by-case, no visa sponsorship
        return 0.15, "outside India (no visa sponsorship)"
    if any(c in loc for c in TIER1_INDIA_CITIES):
        return 1.0, "based in Pune/Noida (JD-preferred hub)"
    if any(c in loc for c in TIER1_INDIA_OTHER):
        return 0.75, "based in a welcomed Tier-1 India city"
    if signals.get("willing_to_relocate"):
        return 0.55, "elsewhere in India but open to relocation"
    return 0.35, "elsewhere in India, relocation preference unclear"


def experience_fit(yoe):
    # Smooth peak at 7 yrs (center of the 6-8 "ideal" band), tapering
    # outside the 5-9 acceptable band. Continuous, so it contributes to
    # score resolution rather than bucketing many candidates together.
    sigma = 2.6
    return math.exp(-((yoe - 7.0) ** 2) / (2 * sigma * sigma))


def tenure_stability(career_history):
    """Reward 3+ year stints; penalize rapid senior-title hopping."""
    if not career_history:
        return 0.5, False
    recent = sorted(career_history, key=lambda r: r.get("start_date") or "", reverse=True)[:3]
    durations = [(r.get("duration_months") or 0) for r in recent]
    avg = sum(durations) / len(durations) if durations else 0
    titles = " ".join(r.get("title", "").lower() for r in recent)
    escalating_senior = has_any(titles, SENIOR_TITLE_KW)
    title_chaser = avg < 18 and escalating_senior and len(career_history) >= 3
    # Smooth, continuous: asymptotically approaches 1.0 as avg tenure grows,
    # never saturates exactly, preserving score resolution across candidates.
    score = 1 - math.exp(-avg / 22.0)
    return score, title_chaser


def stale_ic_check(current_title, career_history):
    """Senior/staff/lead title who hasn't written production code recently.
    Proxy: current title is senior/manager-ish AND none of the last ~2 roles'
    titles contain an IC hint (engineer/scientist/developer)."""
    t = current_title.lower()
    if not has_any(t, ["director", "vp", "head of", "engineering manager", "manager"]):
        return False
    recent = sorted(career_history, key=lambda r: r.get("start_date") or "", reverse=True)[:2]
    recent_titles = " ".join(r.get("title", "").lower() for r in recent)
    return not has_any(recent_titles, IC_HINT_KW)


def behavioral_multiplier(signals):
    m = 1.0
    reasons = []

    if signals.get("open_to_work_flag"):
        m += 0.05
    else:
        m -= 0.05

    last_active = parse_date(signals.get("last_active_date"))
    if last_active:
        days = (TODAY - last_active).days
        if days <= 14:
            m += 0.08
        elif days <= 30:
            m += 0.04
        elif days <= 90:
            pass
        elif days <= 180:
            m -= 0.10
            reasons.append(f"inactive {days}d")
        else:
            m -= 0.25
            reasons.append(f"inactive {days}d (largely unreachable)")

    rr = signals.get("recruiter_response_rate")
    if rr is not None:
        m += (rr - 0.5) * 0.2

    icr = signals.get("interview_completion_rate")
    if icr is not None:
        m += (icr - 0.5) * 0.08

    np_days = signals.get("notice_period_days")
    if np_days is not None:
        if np_days <= 30:
            m += 0.05
        elif np_days <= 60:
            pass
        elif np_days <= 90:
            m -= 0.05
        else:
            m -= 0.10
            reasons.append(f"{np_days}d notice period")

    trust_bonus = 0.0
    if signals.get("verified_email"):
        trust_bonus += 0.01
    if signals.get("verified_phone"):
        trust_bonus += 0.01
    if signals.get("linkedin_connected"):
        trust_bonus += 0.01
    m += trust_bonus

    pc = signals.get("profile_completeness_score")
    if pc is not None:
        m += (pc - 70) / 100 * 0.05

    m = max(0.5, min(1.2, m))
    return m, reasons


def score_candidate(candidate):
    profile = candidate["profile"]
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    evidence_text = extract_evidence_text(candidate)
    current_title = profile.get("current_title", "")
    title_lc = current_title.lower()
    yoe = float(profile.get("years_of_experience", 0) or 0)

    # --- honeypot screen -----------------------------------------------
    hp_score, hp_reasons = detect_honeypot(candidate, evidence_text)
    if hp_score >= 3:
        return None  # dropped from candidate pool entirely

    # --- title relevance --------------------------------------------------
    if has_any(title_lc, TITLE_NEGATIVE):
        title_score = 0.05
        title_note = f'title "{current_title}" is not a technical/ML role'
    elif has_any(title_lc, TITLE_STRONG):
        title_score = 1.0
        title_note = f'title "{current_title}" directly matches the ranking/retrieval/ML-IC profile'
    elif has_any(title_lc, TITLE_MEDIUM):
        title_score = 0.6
        title_note = f'title "{current_title}" is adjacent-technical'
    else:
        title_score = 0.3
        title_note = f'title "{current_title}" has unclear technical relevance'

    # --- production evidence in career-history text -----------------------
    ev_counts = {
        "embeddings": count_any(evidence_text, EVIDENCE_EMBEDDINGS),
        "vectordb": count_any(evidence_text, EVIDENCE_VECTORDB),
        "ranking": count_any(evidence_text, EVIDENCE_RANKING),
        "retrieval": count_any(evidence_text, EVIDENCE_RETRIEVAL),
        "eval": count_any(evidence_text, EVIDENCE_EVAL),
        "pre_llm_ml": count_any(evidence_text, EVIDENCE_PRE_LLM_ML),
        "llm": count_any(evidence_text, EVIDENCE_LLM),
    }
    raw_evidence = (
        ev_counts["embeddings"] * 1.3 + ev_counts["vectordb"] * 1.3 +
        ev_counts["ranking"] * 1.1 + ev_counts["retrieval"] * 1.1 +
        ev_counts["eval"] * 1.0 + ev_counts["pre_llm_ml"] * 0.7 +
        ev_counts["llm"] * 0.5
    )
    # Smooth saturating curve (never hits exactly 1.0) so strong candidates
    # keep differentiating instead of all capping out together.
    evidence_score = 1 - math.exp(-raw_evidence / 4.0)
    strongest_evidence = max(
        [k for k in ev_counts if ev_counts[k] > 0],
        key=lambda k: ev_counts[k], default=None
    )

    # LangChain-only recent AI experience without pre-LLM ML production exp
    langchain_only = (
        ev_counts["llm"] > 0 and ev_counts["pre_llm_ml"] == 0 and
        ev_counts["ranking"] == 0 and ev_counts["retrieval"] == 0 and
        ev_counts["vectordb"] == 0 and ev_counts["embeddings"] == 0
    )

    # --- trust-weighted skills ---------------------------------------------
    skills = candidate.get("skills", [])
    trusted = [s for s in skills if skill_is_trusted(s, evidence_text)]
    relevant_kw = set(
        EVIDENCE_EMBEDDINGS + EVIDENCE_VECTORDB + EVIDENCE_RANKING +
        EVIDENCE_RETRIEVAL + EVIDENCE_LLM + EVIDENCE_EVAL
    )
    trusted_relevant = [
        s for s in trusted
        if any(kw.strip() in s.get("name", "").lower() for kw in relevant_kw)
        or s.get("name", "").lower() in evidence_text
    ]
    skills_score = 1 - math.exp(-len(trusted_relevant) / 3.5)
    untrusted_ai_skill_count = sum(
        1 for s in skills
        if s not in trusted and any(kw.strip() in s.get("name", "").lower() for kw in relevant_kw)
    )

    # --- experience fit ------------------------------------------------
    exp_score = experience_fit(yoe)

    # --- location fit ----------------------------------------------------
    loc_score, loc_note = location_fit(profile, signals)

    # --- education (minor) ------------------------------------------------
    tiers = [e.get("tier") for e in candidate.get("education", [])]
    if "tier_1" in tiers:
        edu_score = 1.0
    elif "tier_2" in tiers:
        edu_score = 0.75
    elif tiers:
        edu_score = 0.5
    else:
        edu_score = 0.5

    # --- stability / tenure ------------------------------------------------
    stability_score, title_chaser = tenure_stability(career)

    # --- weighted base fit ------------------------------------------------
    base_fit = (
        0.28 * title_score +
        0.27 * evidence_score +
        0.15 * skills_score +
        0.10 * exp_score +
        0.10 * loc_score +
        0.05 * edu_score +
        0.05 * stability_score
    )

    # --- disqualifier-style penalties (multiplicative) ---------------------
    penalty = 1.0
    penalty_notes = []

    industries = {r.get("industry", "").lower() for r in career}
    if industries and industries.issubset(RESEARCH_ONLY_INDUSTRIES) or has_any(title_lc, RESEARCH_TITLE_KW):
        if not (ev_counts["ranking"] or ev_counts["retrieval"] or ev_counts["vectordb"]):
            penalty *= 0.15
            penalty_notes.append("pure research background, no production deployment evidence")

    companies = {r.get("company", "").lower() for r in career} | {profile.get("current_company", "").lower()}
    if companies and companies.issubset(CONSULTING_FIRMS):
        penalty *= 0.25
        penalty_notes.append("entire career at consulting/services firms, no product-company experience")

    if has_any(title_lc, CV_SPEECH_ROBOTICS_KW) or has_any(evidence_text, CV_SPEECH_ROBOTICS_KW):
        if not has_any(evidence_text, NLP_IR_KW):
            penalty *= 0.3
            penalty_notes.append("CV/speech/robotics background without NLP/IR exposure")

    if langchain_only:
        penalty *= 0.4
        penalty_notes.append("AI experience appears limited to recent LangChain/LLM-API work with no pre-LLM production ML")

    if stale_ic_check(current_title, career):
        penalty *= 0.5
        penalty_notes.append("senior/management title with no recent hands-on IC role")

    if title_chaser:
        penalty *= 0.6
        penalty_notes.append("rapid title escalation across short stints (possible title-chasing)")

    if signals.get("github_activity_score", -1) == -1 and not candidate.get("certifications") and yoe >= 5:
        penalty *= 0.9
        penalty_notes.append("no external validation signal (no GitHub, no certifications)")

    if untrusted_ai_skill_count >= 4 and evidence_score < 0.2:
        penalty *= 0.35
        penalty_notes.append("many AI/ML skills listed with no endorsements/usage/career corroboration (keyword-stuffing pattern)")

    fit_score = base_fit * penalty

    # --- behavioral multiplier ------------------------------------------------
    behavior_mult, behavior_notes = behavioral_multiplier(signals)
    # Not clipped here — clipping collapses many strong candidates to an
    # identical ceiling. Final normalization (relative to the candidate
    # pool's max) happens once, after scoring everyone, in main().
    raw_score = max(0.0, fit_score * behavior_mult)

    return {
        "candidate_id": candidate["candidate_id"],
        "score": raw_score,
        "title_note": title_note,
        "loc_note": loc_note,
        "strongest_evidence": strongest_evidence,
        "penalty_notes": penalty_notes,
        "behavior_notes": behavior_notes,
        "current_title": current_title,
        "current_company": profile.get("current_company", ""),
        "yoe": yoe,
        "trusted_relevant_skills": [s["name"] for s in trusted_relevant][:4],
        "notice_period": signals.get("notice_period_days"),
        "response_rate": signals.get("recruiter_response_rate"),
        "location": profile.get("location", ""),
    }


def build_reasoning(rank, r):
    facts = []
    facts.append(f'{r["current_title"]} at {r["current_company"]} ({r["yoe"]:.1f} yrs exp)')
    if r["strongest_evidence"]:
        label = {
            "embeddings": "embeddings-based retrieval work",
            "vectordb": "vector DB / hybrid search production experience",
            "ranking": "ranking/recommendation system work",
            "retrieval": "search/retrieval experience",
            "eval": "ranking-evaluation framework experience",
            "pre_llm_ml": "pre-LLM-era applied ML experience",
            "llm": "LLM/RAG experience",
        }[r["strongest_evidence"]]
        facts.append(label)
    if r["trusted_relevant_skills"]:
        facts.append("corroborated skills: " + ", ".join(r["trusted_relevant_skills"]))
    facts.append(r["loc_note"])
    if r["response_rate"] is not None:
        facts.append(f'recruiter response rate {r["response_rate"]:.0%}')
    if r["notice_period"] is not None:
        facts.append(f'{r["notice_period"]}d notice')

    concerns = list(r["penalty_notes"]) + list(r["behavior_notes"])

    # Label reflects the candidate's actual (normalized) score, not just
    # rank position — a candidate shouldn't get "Strong fit" language
    # purely for landing in the top N of a small or unrepresentative
    # sample when the underlying score is weak.
    if r["score"] >= 0.75:
        lead = "Strong fit — "
    elif r["score"] >= 0.55:
        lead = "Solid fit — "
    elif r["score"] >= 0.35:
        lead = "Moderate fit — "
    else:
        lead = "Weak/low-confidence fit — "

    text = lead + "; ".join(facts[:4]) + "."
    if concerns:
        text += " Concern: " + concerns[0] + "."
    text = text[:280]
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()

    scored = []
    dropped_honeypots = 0
    total = 0

    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            candidate = json.loads(line)
            result = score_candidate(candidate)
            if result is None:
                dropped_honeypots += 1
                continue
            scored.append(result)

    if scored:
        max_score = max(r["score"] for r in scored)
        if max_score > 0:
            for r in scored:
                r["score"] = round(r["score"] / max_score * 0.999, 4)

    scored.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    top = scored[: args.top_k]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, r in enumerate(top, start=1):
            reasoning = build_reasoning(i, r)
            writer.writerow([r["candidate_id"], i, f'{r["score"]:.4f}', reasoning])

    print(f"Processed {total} candidates.", file=sys.stderr)
    print(f"Dropped {dropped_honeypots} as suspected honeypots.", file=sys.stderr)
    print(f"Wrote top {len(top)} to {args.out}.", file=sys.stderr)


if __name__ == "__main__":
    main()
