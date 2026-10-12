"""Opt-in necessary-evidence vetoes for an already proposed D5 LLM answer.

This is not a source classifier. Mentions include inactive, future and other
people's sources. We require the interpreter to account for them, not to count
them as active. Passing these finite checks is not a semantic guarantee.
"""
from __future__ import annotations

import re
from typing import Mapping, Sequence

VERSION = "d5-llm-grounding-v3.1"

_PATTERNS = {
    "D5_SALARY": r"\b(?:salar\w*|wages?|pay ?cheques?|pay ?checks?|pay ?packets?|payroll|employment (?:pay|income)|employ(?:ed|er|ee)\w*|paid job|job (?:pay|income)|work (?:pay|income)|shift(?:s)? (?:pay|earnings))\b",
    "D5_SELF_EMPLOYMENT": r"\b(?:self[- ]?employ\w*|freelanc\w*|clients?|invoic\w*|sole trader|business (?:income|earnings|profit)|(?:my|our|own) (?:small )?business|working for myself|contract work|customers?)\b",
    "D5_PENSION_PAYMENTS": r"\b(?:pensions?|retirement (?:pay|income|payments))\b",
    "D5_BENEFITS_SUPPORT": r"\b(?:benefits?|universal credit|welfare|maintenance|alimony|allowance|regular (?:financial )?(?:help|support|assistance)|financial support|support payments?|child support|tax credits?)\b|\b(?:mum|mom|dad|mother|father|parents?|family|partner|friend|daughter|son|sister|brother|relative|aunt|uncle)\b[^.;!?]{0,55}?\b(?:sends?|gives?|transfers?|pays?|covers?|supports?|helps?)\b",
    "D5_WITHDRAWALS": r"\b(?:savings?|saved (?:money|cash)|withdraw\w*|drawdowns?|draw(?:ing|n)? (?:down|from)|dip(?:ping)? into|investments?|investment fund|fund units|shares|capital|reserves?)\b",
    "D5_OTHER_REGULAR": r"\b(?:rent(?:al)? (?:income|receipts?|payments?)|rent (?:from|i receive|received|paid to me)|tenants?|lodgers?|dividends?|interest|royalt\w*|annuities|annuity|licen[cs]e (?:fees|payments?|income)|lease payments?|(?:let|letting) out|rent(?:ing|s)? out)\b",
    "D5_NO_REGULAR": r"\b(?:no|without|don't have|do not have|haven't got|have not got) (?:any |a )?(?:(?:current|regular|recurring|steady|ongoing) )*(?:sources?(?: of income)?|income|funding)\b|^none[.!]?\s*$",
}
PATTERNS = {key: re.compile(value, re.I) for key, value in _PATTERNS.items()}
_COST_USE = re.compile(r"\b(?:live|living|lived) on\b|\b(?:pays?|paying|paid|covers?|covering|funds?|funding|spend|spending|use|using|goes?|going|meets?|meeting|withdraw\w*|draw\w*|transfers?|sends?|gives?|supports?)\b.{0,120}\b(?:bills?|costs?|expenses?|food|groceries|rent|heating|utilities|shopping|council tax|outgoings|essentials)\b", re.I)
_NEGATIVE = re.compile(r"\b(?:not|never|don't|do not|doesn't|does not|isn't|is not|no longer|unused|untouched|stay saved|stays saved|kept saved|reinvest\w*)\b", re.I)
_FUTURE_PAST = re.compile(r"\b(?:will|would|could|might|next (?:month|year|week)|future|used to|previously|formerly|stopped|ended|last year|last month)\b", re.I)
_FUTURE_EVIDENCE = re.compile(r"\b(?:will|would|could|might|future|next (?:month|year|week)|not started yet|starts? next)\b", re.I)
_PAST_EVIDENCE = re.compile(r"\b(?:used to|previously|formerly|stopped|ended|last year|last month|in the past)\b", re.I)
_OTHER = re.compile(r"\b(?:his|her|their|someone else's|my (?:mother|father|brother|sister|friend|partner|mum|dad)'s)\b", re.I)
_OTHER_SUBJECT = re.compile(r"^(?:my (?:mother|father|brother|sister|friend|partner|mum|dad|parents?)|he|she|they)\b", re.I)
_RECIPIENT = re.compile(r"\b(?:to me|to us|supports? me|supports? us|my (?:own )?(?:bills?|costs?|expenses?|food|groceries|rent|heating)|our (?:bills?|costs?|expenses?|food|groceries|rent|heating))\b", re.I)
_UNCERTAIN = re.compile(r"\b(?:not sure|unsure|uncertain|don't know|do not know|haven't checked|have not checked|not checked|can't tell|cannot tell|haven't worked out|have not worked out|maybe|perhaps|probably|might be)\b", re.I)
_AMOUNT_ONLY = re.compile(r"\b(?:how much|the amount|amount next|whether .{0,45}enough|if .{0,45}enough)\b", re.I)
_UNCHANGED_CAPITAL = re.compile(r"\b(?:leave|keep|leaving|keeping)\b.{0,45}\b(?:capital|principal|original (?:savings|amount))\b|\b(?:capital|principal|original (?:savings|amount))\b.{0,35}\b(?:in place|intact|unchanged)\b", re.I)
_NO_USE = re.compile(r"\b(?:unused|untouched|stay saved|stays saved|kept saved|reinvest\w*)\b|\b(?:don't|do not|doesn't|does not|never|not|no longer)\b.{0,25}\b(?:use|used|using|pay|pays|paying|fund|funds|funding|spend|spending|draw|drawing|withdraw\w*|cover|covers|covering)\b", re.I)
_CAPITAL_USE = re.compile(r"\b(?:withdraw\w*|drawdowns?|draw(?:ing|n)? (?:down|from)|dip(?:ping)? into|sell|selling|sold|sale proceeds|take money (?:out of|from)|use up|spend(?:ing)? (?:my |our |the )?(?:own )?(?:savings|saved money|capital)|transfers? (?:my |our |the )?(?:own )?saved money)\b", re.I)
_PENSION_HOLDING = re.compile(r"\bpension (?:pot|balance|fund|holdings|account|savings)\b", re.I)
_PENSION_PAYMENT = re.compile(r"\bpension (?:payments?|income)\b|\b(?:receive|receiving|get|getting|paid)\b.{0,25}\bpension\b", re.I)


def _enclosing_clauses(original: str, evidence: str):
    """Trusted original context prevents a quote from cropping a modifier."""
    for match in re.finditer(re.escape(evidence), original):
        left = 0
        right = len(original)
        content_end = match.end() - (len(evidence)-len(evidence.rstrip(".;!?\n")))
        for boundary in re.finditer(r"[.;!?\n]+|\bbut\b|\bhowever\b", original, re.I):
            if boundary.end() <= match.start(): left = boundary.end()
            elif boundary.start() >= content_end:
                right = boundary.start()
                break
        yield _text(original[left:right].strip())


def _text(value: str) -> str:
    return value.replace("’", "'")


def only_nonmaterial_uncertainty(original: str) -> bool:
    """Necessary scoped exemption, never an answer: every hedge is future/amount.

    Generic 'not sure which source pays' cannot pass. Call only after grounding
    and structural relation checks have supported the same proposed option.
    """
    uncertain = [c for c in re.split(r"[.;!?]+|\bbut\b", _text(original)) if _UNCERTAIN.search(c)]
    return bool(uncertain) and all(_AMOUNT_ONLY.search(c) or (
        re.search(r"\b(?:future|next (?:year|month|week))\b", c, re.I)
        and not _COST_USE.search(c)
        and not re.search(r"\b(?:now|current|currently|today)\b", c, re.I)) for c in uncertain)


def grounding_violations(original: str, facts: Sequence[Mapping[str, str]], option: str) -> tuple[str, ...]:
    """Return veto reasons only. Never return an option, bucket or repaired fact."""
    text = _text(original)
    issues = []
    for fact in facts:
        category = fact.get("source_category", "")
        evidence = fact.get("evidence", "")
        normalized = _text(evidence)
        pattern = PATTERNS.get(category)
        if not evidence or evidence not in original or pattern is None or not pattern.search(normalized):
            issues.append("source_category_not_supported_by_its_evidence")
            continue
        positive = fact.get("owner") == "self" and fact.get("time") == "current" and fact.get("pays_living_costs") == "yes"
        source_contexts = [context for anchor in pattern.finditer(normalized)
                           for context in _enclosing_clauses(normalized, anchor.group())]
        # Inactive assertions need their own evidence too. Otherwise a model
        # can acknowledge a second active source while labelling it future/no
        # to conceal it from the option calculation.
        if fact.get("owner") == "other" and not all(_OTHER.search(c) or _OTHER_SUBJECT.search(c) for c in source_contexts):
            issues.append("other_owner_relation_not_supported")
        if fact.get("owner") == "unknown" and not _UNCERTAIN.search(normalized):
            issues.append("unknown_owner_relation_not_supported")
        if fact.get("time") == "future" and not all(_FUTURE_EVIDENCE.search(c) for c in source_contexts):
            issues.append("future_relation_not_supported")
        if fact.get("time") == "past" and not all(_PAST_EVIDENCE.search(c) for c in source_contexts):
            issues.append("past_relation_not_supported")
        if fact.get("time") == "unknown" and not _UNCERTAIN.search(normalized):
            issues.append("unknown_time_relation_not_supported")
        if fact.get("pays_living_costs") == "no" and category != "D5_NO_REGULAR" and not all(
            _NO_USE.search(c) or _UNCHANGED_CAPITAL.search(c)
            or fact.get("time") == "future" and _FUTURE_EVIDENCE.search(c)
            or fact.get("time") == "past" and _PAST_EVIDENCE.search(c) for c in source_contexts):
            issues.append("inactive_use_relation_not_supported")
        if fact.get("pays_living_costs") == "unknown" and not _UNCERTAIN.search(normalized):
            issues.append("unknown_use_relation_not_supported")
        if not positive:
            continue
        # A short direct bounded answer can rely on the question's current-use
        # premise. Longer replies must supply a use relationship in evidence.
        terse = normalized.strip() == text.strip() and bool(re.fullmatch(
            r"(?:my |our )?(?:salary|wages|pension payments|benefits|universal credit|regular support|withdrawals(?: from savings)?|rental income|rent income|interest income|dividends)[.!]?", normalized.strip(), re.I))
        if not terse and not _COST_USE.search(normalized):
            issues.append("current_living_cost_use_not_supported_by_evidence")
        if category == "D5_BENEFITS_SUPPORT" and _OTHER_SUBJECT.search(normalized) and not _RECIPIENT.search(normalized):
            issues.append("received_support_recipient_not_supported")
        if category == "D5_WITHDRAWALS" and (not _CAPITAL_USE.search(normalized)
                or re.search(r"\b(?:interest|dividends?)\b", normalized, re.I)):
            issues.append("capital_use_not_supported_by_evidence")
        if category == "D5_PENSION_PAYMENTS" and _PENSION_HOLDING.search(normalized) and not _PENSION_PAYMENT.search(normalized):
            issues.append("pension_holding_is_not_payment_evidence")
        for context in _enclosing_clauses(original, evidence):
            outside = context.replace(normalized.rstrip(".;!?\n"), "", 1)
            if (_NEGATIVE.search(outside) or _FUTURE_PAST.search(outside)
                    or _UNCERTAIN.search(outside) or _OTHER.search(outside)
                    or category != "D5_BENEFITS_SUPPORT" and _OTHER_SUBJECT.search(context)):
                issues.append("quoted_fact_crops_material_original_context")
        # Narrow contradictions near the proposed source are conservative
        # vetoes, not general negation/tense resolution. Complex cases abstain.
        for anchor in pattern.finditer(normalized):
            near = normalized[max(0, anchor.start()-42):min(len(normalized), anchor.end()+65)]
            if _FUTURE_PAST.search(near) or _NEGATIVE.search(near):
                issues.append("positive_fact_has_unresolved_time_or_negation_context")
            if category != "D5_BENEFITS_SUPPORT" and _OTHER.search(near):
                issues.append("positive_fact_has_other_owner_context")
            if category != "D5_BENEFITS_SUPPORT" and _OTHER_SUBJECT.search(normalized):
                issues.append("positive_fact_has_other_owner_context")
    # Every recognisable source mention needs an explicitly represented fact,
    # even if that fact says other/past/future/no/unknown. Never treat the word
    # itself as a contributor. This catches omitted sources and omitted doubt.
    for category, pattern in PATTERNS.items():
        if category == "D5_NO_REGULAR":
            continue
        for mention in pattern.finditer(text):
            represented = False
            for fact in facts:
                if fact.get("source_category") != category:
                    continue
                span = _text(fact.get("evidence", ""))
                # All exact occurrences are eligible, without inventing an
                # offset supplied by the model. The whole source must be in it.
                start = text.find(span) if span else -1
                while start >= 0:
                    if start <= mention.start() and mention.end() <= start+len(span):
                        represented = True
                        break
                    start = text.find(span, start+1)
                if represented:
                    break
            if not represented:
                issues.append("full_reply_source_mention_unaccounted_for")
    # A model cannot erase expressed current-use uncertainty merely by leaving
    # its missing_information list empty. Amount-only uncertainty is different.
    for clause in re.split(r"[.;!?]+|\bbut\b", text):
        if _UNCERTAIN.search(clause) and not _AMOUNT_ONLY.search(clause) and (
                not _FUTURE_PAST.search(clause) or _COST_USE.search(clause)):
            if any(p.search(clause) for k,p in PATTERNS.items() if k != "D5_NO_REGULAR"):
                issues.append("current_source_uncertainty_not_resolved")
    return tuple(dict.fromkeys(issues))
