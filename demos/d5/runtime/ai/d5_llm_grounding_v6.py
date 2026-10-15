"""Opt-in necessary-evidence vetoes for an already proposed D5 LLM answer.

This is not a source classifier. Mentions include inactive, future and other
people's sources. We require the interpreter to account for them, not to count
them as active. Passing these finite checks is not a semantic guarantee.

Version 6 adds finite wording variants and narrowly scoped denial handling.
It never normalises or repairs the quoted evidence or supplies a category.
"""
from __future__ import annotations

import re
from typing import Mapping, Sequence

VERSION = "d5-llm-grounding-v6.1"

_PATTERNS = {
    "D5_SALARY": r"\b(?:salar\w*|wages?|wags|waegs|salery|sallary|pay ?cheques?|pay ?checks?|pay ?packets?|payroll|(?<!self-)(?<!self )employment (?:pay|income|earnings|payments)|(?<!self-)(?<!self )employ(?:s|ed|er|ee|ing)\w*|paid job|job (?:pay|income)|work (?:pay|income)|shift(?:s)? (?:pay|earnings))\b",
    "D5_SELF_EMPLOYMENT": r"\b(?:self[- ]?employ\w*|freelanc\w*|clients?|invoic\w*|sole trader|business (?:income|earnings|profit)|(?:my|our|own) (?:small )?business|working for myself|contract work|customers?)\b",
    "D5_PENSION_PAYMENTS": r"\b(?:pensions?|penssion|pention|pensoin|retirement (?:pay|income|payments))\b",
    "D5_BENEFITS_SUPPORT": r"\b(?:benefits?|benifits?|universal credit|welfare|maintenance|alimony|allowance|regular (?:financial )?(?:help|support|assistance)|financial support|support payments?|child support|tax credits?)\b|\b(?:mum|mom|dad|mother|father|parents?|family|partner|friends?|daughters?|sons?|sisters?|brothers?|relatives?|aunts?|uncles?)\b[^.;!?]{0,55}?\b(?:sends?|gives?|transfers?|pays?|covers?|supports?|helps?|provides?)\b",
    "D5_WITHDRAWALS": r"\b(?:savings?|saved (?:money|cash|balance)|withdraw\w*|drawdowns?|draw(?:ing|n)? (?:down|from)|draw(?:ing)? money out of|dip(?:ping)? into|investments?|investment fund|fund units|shares|capital|reserves?)\b",
    "D5_OTHER_REGULAR": r"\b(?:rent(?:al)? (?:income|receipts?|payments?)|rent (?:from|i receive|received|paid to me)|tenants?|lodgers?|dividends?|dividents?|interest|royalt\w*|annuities|annuity|licen[cs]e (?:fees|payments?|income)|lease payments?|(?:let|letting) out|rent(?:ing|s)? out)\b",
    "D5_NO_REGULAR": r"\b(?:no|without|don't have|do not have|haven't got|have not got) (?:any |a )?(?:(?:current|regular|recurring|steady|ongoing) )*(?:sources?(?: of income)?|income|funding|earnings|receipts)\b|^none[.!]?\s*$",
}
PATTERNS = {key: re.compile(value, re.I) for key, value in _PATTERNS.items()}
_COST_WORDS = r"(?:bills?|costs?|expenses?|food|groceries|rent|heating|utilities|shopping|council tax|outgoings|essentials|necessities|expenditure|upkeep|household needs|daily needs|day[- ]to[- ]day needs)"
_USE_VERBS = r"(?:pays?|paying|paid|covers?|covering|covered|funds?|funding|funded|finances?|financing|financed|spend|spending|spent|use|used|using|goes?|going|buys?|buying|meets?|meeting|met|withdraw\w*|draw\w*|transfers?|sends?|gives?|supports?|provides?)"
_COST_USE = re.compile(
    r"\b(?:live|living|lived) (?:entirely )?on\b|\b" + _USE_VERBS + r"\b.{0,120}\b" + _COST_WORDS + r"\b|"
    r"\b" + _COST_WORDS + r"\b.{0,80}\b(?:paid|funded|financed|covered)\b|"
    r"\b(?:keep|keeps|keeping) (?:a |the )?roof over (?:my|our) heads?\b|"
    r"\b(?:put|puts|putting) food on (?:my|our|the) table\b", re.I)
_NEGATIVE = re.compile(r"\b(?:not|never|don't|do not|doesn't|does not|isn't|is not|no longer|unused|untouched|stay saved|stays saved|kept saved|reinvest\w*)\b", re.I)
_FUTURE_PAST = re.compile(r"\b(?:will|would|could|might|next (?:month|year|week)|future|used to|previously|formerly|stopped|ended|last year|last month)\b", re.I)
_FUTURE_EVIDENCE = re.compile(r"\b(?:will|would|could|might|future|next (?:month|year|week)|not started yet|starts? next)\b", re.I)
_PAST_EVIDENCE = re.compile(r"\b(?:used to|previously|formerly|stopped|ended|last year|last month|in the past)\b", re.I)
_OTHER = re.compile(r"\b(?:his|her|their|someone else's|my (?:mother|father|brother|sister|friend|partner|mum|dad|daughter|son|relative|aunt|uncle|neighbour|neighbor)'s)\b", re.I)
_OTHER_SUBJECT = re.compile(r"^(?:my (?:mother|father|brother|sister|friend|partner|mum|dad|parents?|daughter|son|relative|aunt|uncle|neighbour|neighbor)|he|she|they)\b", re.I)
_RECIPIENT = re.compile(r"\b(?:to me|to us|supports? me|supports? us|my (?:own )?(?:household |daily |everyday |living )?(?:bills?|costs?|expenses?|food|groceries|rent|heating|needs|necessities)|our (?:household |daily |everyday |living )?(?:bills?|costs?|expenses?|food|groceries|rent|heating|needs|necessities))\b", re.I)
_UNCERTAIN = re.compile(r"\b(?:not sure|unsure|uncertain|don't know|do not know|haven't checked|have not checked|not checked|can't tell|cannot tell|haven't worked out|have not worked out|maybe|perhaps|probably|might be)\b", re.I)
_AMOUNT_ONLY = re.compile(r"\b(?:how much|the amount|amount next|whether .{0,45}enough|if .{0,45}enough)\b", re.I)
_UNCHANGED_CAPITAL = re.compile(r"\b(?:leave|keep|leaving|keeping)\b.{0,45}\b(?:capital|principal|original (?:savings|amount))\b|\b(?:capital|principal|original (?:savings|amount))\b.{0,35}\b(?:in place|intact|unchanged)\b", re.I)
_NO_USE = re.compile(r"\b(?:unused|untouched|stay saved|stays saved|kept saved|reinvest\w*)\b|\b(?:don't|do not|doesn't|does not|never|not|no longer)\b.{0,25}\b(?:use|used|using|pay|pays|paying|paid|fund|funds|funding|funded|finance|finances|financing|financed|spend|spending|spent|draw|drawing|withdraw\w*|cover|covers|covering|covered|meet|meets|meeting)\b", re.I)
_CAPITAL_USE = re.compile(r"\b(?:withdraw\w*|drawdowns?|draw(?:ing|n)? (?:down|from)|draw(?:ing)? money out of|dip(?:ping)? into|sell|selling|sold|sale proceeds|take money (?:out of|from)|use up|spend(?:ing)? (?:my |our |the )?(?:own )?(?:savings|saved money|saved balance|capital)|transfers? (?:my |our |the )?(?:own )?saved (?:money|balance))\b", re.I)
_PENSION_HOLDING = re.compile(r"\b(?:pension|penssion|pention|pensoin) (?:pot|balance|fund|holdings|account|savings)\b", re.I)
_PENSION_PAYMENT = re.compile(r"\b(?:pension|penssion|pention|pensoin) (?:payments?|income|instalments?|receipts)\b|\b(?:receive|receiving|get|getting|paid)\b.{0,25}\b(?:pension|penssion|pention|pensoin)\b", re.I)

# Only direct denial of dividend/interest receipts is scoped here. A denial of
# amount, adequacy or exclusive use ("not dividends alone") is not no-use.
_DENIED_OTHER_RECEIPT = re.compile(
    r"\b(?:not|no)\s+(?:(?:my|our|any)\s+)?(?:dividends?|dividents?|interest(?: income)?)\b"
    r"(?!\s+(?:alone|only|enough|entirely|just|necessarily)\b)", re.I)


def _mask_accounted_receipt_denials(value: str, facts: Sequence[Mapping[str, str]], *, category: str) -> str:
    """Hide only represented, different-category receipt denials from a veto.

    The original exact quotes stay untouched. This mask is solely local
    negation scoping, never interpretation or repaired evidence.
    """
    if category == "D5_OTHER_REGULAR":
        return value
    accounted = [f for f in facts if f.get("source_category") == "D5_OTHER_REGULAR"
                 and f.get("time") == "current" and f.get("pays_living_costs") == "no"
                 and _DENIED_OTHER_RECEIPT.search(_text(f.get("evidence", "")))]
    if not accounted:
        return value
    def replace(match):
        phrase = match.group()
        if any(phrase in _text(f.get("evidence", "")) for f in accounted):
            return " " * len(phrase)
        return phrase
    return _DENIED_OTHER_RECEIPT.sub(replace, value)


def _direct_source_denial(category: str, context: str) -> bool:
    """Necessary no-use wording for specifically denied other receipts only."""
    if category != "D5_OTHER_REGULAR":
        return False
    mentions = list(PATTERNS[category].finditer(context))
    denials = list(_DENIED_OTHER_RECEIPT.finditer(context))
    return bool(mentions) and all(any(d.start() <= m.start() and m.end() <= d.end()
                                    for d in denials) for m in mentions)


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


# These deliberately strong local premises are used only to veto an existing
# candidate. They never supply a replacement option or determine a bucket.
_SELF_COST_OR_RECEIPT = re.compile(r"\b(?:i|we|my|our|to me|to us)\b", re.I)
_CURRENT_USE = re.compile(
    r"\b(?:pay|pays|paying|cover|covers|covering|fund|funds|funding|finance|finances|financing|"
    r"use|using|spend|spending|withdraw|withdraws|withdrawing|draw|draws|drawing|buy|buys|buying|"
    r"meet|meets|meeting|live|living|send|sends|sending|give|gives|giving|"
    r"transfer|transfers|transferring|support|supports|supporting|provide|provides|providing)\b|"
    r"\b(?:is|are) (?:paid|covered|funded|financed)\b", re.I)
_NON_CURRENT_PROOF = re.compile(
    r"\b(?:will|would|could|might|may|future|next (?:week|month|year|term)|used to|previously|"
    r"formerly|yesterday|last (?:week|month|year)|in the past|had|was|were|stopped|ended)\b", re.I)
_ONE_OFF_PROOF = re.compile(r"\b(?:one[- ]off|one[- ]time|occasional|isolated|single (?:gift|payment))\b", re.I)
_DENIED_FOCAL_PROOF = re.compile(r"^(?:with no|without|no)\b", re.I)
_LABELS = {
    "D5_SALARY": r"(?:salary|wages?|employment pay)",
    "D5_SELF_EMPLOYMENT": r"(?:self[- ]employed earnings|self[- ]employment income|freelance earnings)",
    "D5_PENSION_PAYMENTS": r"(?:pension payments?|pension income|retirement payments?)",
    "D5_BENEFITS_SUPPORT": r"(?:benefits?|universal credit|regular support|regular family support)",
    "D5_WITHDRAWALS": r"(?:withdrawals(?: from (?:my |our )?(?:own )?savings)?|savings withdrawals)",
    "D5_OTHER_REGULAR": r"(?:rental income|rent income|dividends?|interest income|royalties)",
}
_LABEL_PATTERN = "(?:" + "|".join(_LABELS.values()) + ")"
_LABEL_LIST = re.compile(r"^(?:(?:my|our) )?" + _LABEL_PATTERN
    + r"(?:\s*(?:,|\+|&|and|/)\s*(?:(?:my|our) )?" + _LABEL_PATTERN + r")*[.!]?$", re.I)


def _financial_focal_clauses(original: str) -> tuple[str, ...]:
    # Split a coordinated source clause only when it has its own personal
    # subject; shared 'my salary and pension payments pay bills' stays whole.
    clauses = [c.strip(" ,") for c in re.split(
        r"[.;!?]+|\b(?:but|however|while|whereas)\b|\band\s+(?=(?:i|we|he|she|they)\b)|"
        r"\band\s+(?=(?:my (?:mother|father|brother|sister|friend|partner|mum|dad|parents?|"
        r"daughter|son|relative|aunt|uncle|neighbour|neighbor)|his|her|their)\b)|"
        r"\s+(?=(?:with no|without)\b)",
        _text(original)) if c.strip(" ,")]
    result = list(clauses)
    for left, right in zip(clauses, clauses[1:]):
        if re.match(r"^(?:it|this (?:income|payment|money)|that (?:income|payment|money))\b", right, re.I):
            result.append(left + ". " + right)
    return tuple(result)


def positive_current_source_categories(original: str) -> frozenset[str]:
    """Strong finite current-use premises for vetoes; never a classifier.

    A named mention is insufficient. Focal evidence must support personal use,
    present actual living-cost funding, and the source's necessary meaning.
    Unsupported, future, negative, donor-only or uncertain mentions add nothing.
    Short unqualified category lists inherit the bounded D5 question premise.
    """
    text = _text(original).strip()
    if _LABEL_LIST.fullmatch(text):
        return frozenset(k for k, pattern in _LABELS.items() if re.search(pattern, text, re.I))
    positive = set()
    for clause in _financial_focal_clauses(text):
        if (_NON_CURRENT_PROOF.search(clause) or _NEGATIVE.search(clause)
                or _DENIED_FOCAL_PROOF.search(clause)
                or _UNCERTAIN.search(clause) or _ONE_OFF_PROOF.search(clause)
                or not _SELF_COST_OR_RECEIPT.search(clause)
                or not _CURRENT_USE.search(clause) or not _COST_USE.search(clause)):
            continue
        for category, pattern in PATTERNS.items():
            if category == "D5_NO_REGULAR" or not pattern.search(clause):
                continue
            if category == "D5_BENEFITS_SUPPORT":
                if (_OTHER.search(clause) or _OTHER_SUBJECT.search(clause)) and not _RECIPIENT.search(clause):
                    continue
            elif _OTHER.search(clause) or _OTHER_SUBJECT.search(clause):
                continue
            if category == "D5_WITHDRAWALS" and not _CAPITAL_USE.search(clause):
                continue
            if category == "D5_PENSION_PAYMENTS" and _PENSION_HOLDING.search(clause) and not _PENSION_PAYMENT.search(clause):
                continue
            positive.add(category)
    return frozenset(positive)


def grounding_violations(original: str, facts: Sequence[Mapping[str, str]], option: str) -> tuple[str, ...]:
    """Return veto reasons only. Never return an option, bucket or repaired fact."""
    text = _text(original)
    issues = []
    active = {f.get("source_category") for f in facts
              if f.get("owner") == "self" and f.get("time") == "current"
              and f.get("pays_living_costs") == "yes"
              and f.get("source_category") not in {None, "D5_NO_REGULAR", "D5_MULTIPLE"}}
    if option == "D5_MULTIPLE" and len(active) < 2:
        issues.append("multiple_lacks_two_current_self_source_facts")
    elif option != "D5_MULTIPLE" and (len(active) >= 2 or len(positive_current_source_categories(original)) >= 2):
        issues.append("single_source_candidate_conflicts_with_distinct_current_sources")
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
            _NO_USE.search(c) or _UNCHANGED_CAPITAL.search(c) or _direct_source_denial(category, c)
            or fact.get("time") == "future" and _FUTURE_EVIDENCE.search(c)
            or fact.get("time") == "past" and _PAST_EVIDENCE.search(c) for c in source_contexts):
            issues.append("inactive_use_relation_not_supported")
        if fact.get("pays_living_costs") == "unknown" and not _UNCERTAIN.search(normalized):
            issues.append("unknown_use_relation_not_supported")
        if category == "D5_NO_REGULAR" and fact.get("time") == "current":
            # A future or another person's absence of income is not the
            # customer's current absence. The negative no-use fact is required
            # by the strict parser; it does not get a positive-use exemption.
            if any(_FUTURE_PAST.search(c) or _OTHER.search(c) or _OTHER_SUBJECT.search(c)
                   for c in source_contexts):
                issues.append("current_no_regular_relation_not_supported")
            for context in _enclosing_clauses(original, evidence):
                outside = context.replace(normalized.rstrip(".;!?\n"), "", 1)
                if _FUTURE_PAST.search(outside) or _OTHER.search(outside) or _OTHER_SUBJECT.search(context):
                    issues.append("quoted_fact_crops_material_original_context")
        if not positive:
            continue
        # A short direct bounded answer can rely on the question's current-use
        # premise. Longer replies must supply a use relationship in evidence.
        terse = normalized.strip() == text.strip() and bool(re.fullmatch(
            r"(?:my |our )?(?:salary|wages|pension payments|benefits|universal credit|regular support|withdrawals(?: from savings)?|rental income|rent income|interest income|dividends)[.!]?", normalized.strip(), re.I))
        if not terse and not _COST_USE.search(normalized):
            issues.append("current_living_cost_use_not_supported_by_evidence")
        if category == "D5_BENEFITS_SUPPORT" and (
                _OTHER_SUBJECT.search(normalized) or _OTHER.search(normalized)) and not _RECIPIENT.search(normalized):
            issues.append("received_support_recipient_not_supported")
        if category == "D5_WITHDRAWALS" and (not _CAPITAL_USE.search(normalized)
                or re.search(r"\b(?:interest|dividends?|dividents?)\b",
                    _mask_accounted_receipt_denials(normalized, facts, category=category), re.I)):
            issues.append("capital_use_not_supported_by_evidence")
        if category == "D5_PENSION_PAYMENTS" and _PENSION_HOLDING.search(normalized) and not _PENSION_PAYMENT.search(normalized):
            issues.append("pension_holding_is_not_payment_evidence")
        for context in _enclosing_clauses(original, evidence):
            outside = context.replace(normalized.rstrip(".;!?\n"), "", 1)
            outside = _mask_accounted_receipt_denials(outside, facts, category=category)
            if (_NEGATIVE.search(outside) or _FUTURE_PAST.search(outside)
                    or _UNCERTAIN.search(outside) or _OTHER.search(outside)
                    or category != "D5_BENEFITS_SUPPORT" and _OTHER_SUBJECT.search(context)):
                issues.append("quoted_fact_crops_material_original_context")
        # Narrow contradictions near the proposed source are conservative
        # vetoes, not general negation/tense resolution. Complex cases abstain.
        for anchor in pattern.finditer(normalized):
            near = normalized[max(0, anchor.start()-42):min(len(normalized), anchor.end()+65)]
            near = _mask_accounted_receipt_denials(near, facts, category=category)
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
