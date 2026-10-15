"""Opt-in necessary-evidence vetoes for an already proposed D5 LLM answer.

This is not a source classifier. Mentions include inactive, future and other
people's sources. We require the interpreter to account for them, not to count
them as active. Passing these finite checks is not a semantic guarantee.

Version 8 scopes adjacent receipt/use evidence, shared predicates and explicit
absence lists. It retains exact evidence and veto-only output.
It never normalises or repairs the quoted evidence or supplies a category.
"""
from __future__ import annotations

import re
from typing import Mapping, Sequence

VERSION = "d5-llm-grounding-v8.1"

_PATTERNS = {
    "D5_SALARY": r"\b(?:salar\w*|wages?|wags|waegs|salery|sallary|pay ?cheques?|pay ?checks?|pay ?packets?|payroll|(?<!self-)(?<!self )employment (?:pay|income|earnings|payments)|(?<!self-)(?<!self )employ(?:s|ed|er|ee|ing)\w*|paid job|job (?:pay|income)|work (?:pay|income)|shift(?:s)? (?:pay|earnings))\b",
    "D5_SELF_EMPLOYMENT": r"\b(?:business|trade) (?:i|we) own\b|\b(?:i|we) (?:run|own) (?:an? |my |our )?(?:[a-z]+[- ]?){0,3}(?:business|trade)\b|\b(?:my|our|own) (?:own )?(?:[a-z]+[- ]?){0,3}(?:business|trade)\b|\b(?:self[- ]?employ\w*|freelanc\w*|clients?|invoic\w*|sole trader|business (?:income|earnings|profit)|(?:my|our|own) (?:small )?business|workin(?:g)? for myself|contract work|customers?)\b",
    "D5_PENSION_PAYMENTS": r"\b(?:pensions?|penson|penssion|pention|pensoin|retirement (?:pay|income|payments))\b",
    "D5_BENEFITS_SUPPORT": r"\b(?:regular|recurring|ongoing) money from (?:my |our )?(?:friends?|family|parents?|mother|father|mum|dad|sisters?|brothers?|partners?|relatives?)\b|\b(?:regular|recurring|ongoing) money sent to (?:me|us) by (?:my |our )?(?:friends?|family|parents?|mother|father|mum|dad|sisters?|brothers?|partners?|relatives?)\b|\b(?:benefits?|benifits?|universal credit|welfare|(?<!-)maintenance(?!-)|alimony|allowance|(?:regular|recurring|ongoing|received) (?:financial )?(?:help|support|assistance)|financial support|support payments?|child support|tax credits?)\b|\b(?:mum|mom|dad|mother|father|grandmother|grandfather|grandparents?|wife|wives|husband|spouse|parents?|family|partner|friends?|daughters?|sons?|nephews?|nieces?|cousins?|sisters?|brothers?|relatives?|aunts?|uncles?)\b[^.;!?]{0,55}?\b(?:send(?:s|ing)?|giv(?:e|es|ing)|transfer(?:s|ring)?|pays?|covers?|supports?|helps?|provides?|contribut(?:e|es|ing))\b",
    "D5_WITHDRAWALS": r"\b(?:sale proceeds|sales of shares|selling units|sell (?:some |my |our )?assets)\b|\b(?:savings?|saved[- ]up (?:money|cash)|saved (?:money|cash|balance)|withdraw\w*|drawdowns?|draw(?:ing|n)? (?:down|from)|draw(?:ing)? money out of|dip(?:ping)? into|investments?|investment fund|fund units|shares|capital|reserves?)\b",
    "D5_OTHER_REGULAR": r"\b(?:rent(?:al)? (?:income|receipts?|payments?)|rent (?:from|i receive|received|paid to me)|tenants?|lodgers?|dividends?|dividents?|interest|royalt\w*|annuities|annuity|licen[cs]e (?:fees|payments?|income)|lease payments?|(?:let|letting) out|rent(?:ing|s)? out)\b",
    "D5_NO_REGULAR": r"\b(?:nothing|no money|no source) (?:regular(?:ly)?|recurring|ongoing) (?:is )?(?:funding|pays|paying|covering|providing money for)\b|\b(?:no|without|don't have|do not have|haven't got|have not got) (?:any |a )?(?:(?:current|regular|recurring|steady|ongoing) )*(?:sources?(?: of income)?|income|funding|earnings|receipts|living[- ]money source)\b|^none[.!]?\s*$",
}
PATTERNS = {key: re.compile(value, re.I) for key, value in _PATTERNS.items()}
_COST_WORDS = r"(?:(?:(?:living|daily|everyday|household|regular|usual|normal|home) )?(?:bills?|costs?|expenses?|food|groceries|meals|everyday stuff|rent|heating|utilities|shopping|council tax|outgoings|essentials|necessities|expenditure|upkeep|stuff|needs)|(?:daily|everyday|ordinary|normal) (?:life|living|living money)|(?:household|home) spending|(?:normal )?home needs|living money|money for living|day[- ]to[- ]day (?:needs|costs|expenses|stuff))"
_USE_VERBS = r"(?:pays?|paying|paid|covers?|covering|covered|funds?|funding|funded|finances?|financing|financed|spend|spending|spent|use|used|using|go(?:es)?|going|buys?|buying|meets?|meeting|met|withdraw\w*|draw\w*|tak(?:e|es|ing)|suppl(?:y|ies|ying)|transfer(?:s|ring)?|send(?:s|ing)?|giv(?:e|es|ing)|supports?|provides?|contribut(?:e|es|ing))"
_COST_USE = re.compile(
    r"\b(?:live|living|lived) (?:entirely )?on\b|\b" + _USE_VERBS + r"\b.{0,120}\b" + _COST_WORDS + r"\b|"
    r"\b" + _COST_WORDS + r"\b.{0,80}\b(?:paid|funded|financed|covered)\b|"
    r"\b(?:keep|keeps|keeping) (?:a |the )?roof over (?:my|our) heads?\b|"
    r"\b(?:put|puts|putting) food on (?:my|our|the) table\b", re.I)
# Bounded contraction variants are evidence of inactivity, not model repair.
# They are checked in the original text; quoted evidence remains unchanged.
_FUTURE_CONTRACTION_TEXT = r"(?:i|we|he|she|it|they)'ll|won'?t"
_NOT_STARTED_TEXT = (
    r"(?:haven'?t|hasn'?t|have not|has not) (?:yet )?(?:started|begun)|"
    r"(?:not yet|yet to) (?:start|begin)"
)
_NOT_RECEIVED_TEXT = r"(?:haven'?t|hasn'?t|have not|has not) (?:yet )?(?:received|got|had)"
_INABILITY_USE_TEXT = r"(?:can'?t|cannot|can not|unable to) (?:currently )?(?:use|spend|draw|withdraw)"
_INABILITY_PAY = re.compile(r"\b(?:can'?t|cannot|can not|unable to) (?:currently )?(?:pay|cover|fund)\b", re.I)
_NEGATIVE = re.compile(r"\b(?:" + _NOT_STARTED_TEXT + r"|" + _NOT_RECEIVED_TEXT + r"|" + _INABILITY_USE_TEXT + r"|won'?t|not|never|don'?t|do not|doesn'?t|does not|isn't|is not|no longer|unused|untouched|stay saved|stays saved|kept saved|reinvest\w*)\b", re.I)
_FUTURE_PAST = re.compile(r"\b(?:" + _NOT_STARTED_TEXT + r"|" + _FUTURE_CONTRACTION_TEXT + r"|will|would|could|might|next (?:month|year|week)|future|used to|previously|formerly|stopped|ended|last year|last month)\b", re.I)
_FUTURE_EVIDENCE = re.compile(r"\b(?:" + _NOT_STARTED_TEXT + r"|" + _FUTURE_CONTRACTION_TEXT + r"|will|would|could|might|future|next (?:month|year|week)|not started yet|starts? next)\b", re.I)
_PAST_EVIDENCE = re.compile(r"\b(?:used to|previously|formerly|stopped|ended|last year|last month|in the past)\b", re.I)
_OTHER = re.compile(r"\b(?:his|her|their|someone else's|my (?:mother|father|brother|sister|friend|partner|wife|wives|husband|spouse|grandmother|grandfather|grandparents?|boss|employer|mum|dad|daughters?|sons?|nephews?|nieces?|cousins?|relative|aunt|uncle|neighbour|neighbor)'s)\b", re.I)
_OTHER_SUBJECT = re.compile(r"^(?:(?:my|our|the|a|an) (?:mother|father|brother|sister|friend|partner|wife|wives|husband|spouse|grandmother|grandfather|grandparents?|mum|dad|parents?|daughters?|sons?|nephews?|nieces?|cousins?|relative|aunt|uncle|neighbour|neighbor)|he|she|they)\b", re.I)
_RECIPIENT = re.compile(r"\b(?:to me|to us|(?:sends?|gives?|transfers?|supports?) (?:me|us)|(?:my|our) (?:own )?(?:household |daily |everyday |living |normal |home )*(?:bills?|costs?|expenses?|food|groceries|rent|heating|needs|necessities))\b", re.I)
_UNCERTAIN = re.compile(r"\b(?:not sure|unsure|uncertain|don't know|do not know|haven't checked|have not checked|not checked|can't tell|cannot tell|haven't worked out|have not worked out|maybe|perhaps|probably|might be)\b", re.I)
_AMOUNT_ONLY = re.compile(r"\b(?:how much|the amount|amount next|whether .{0,45}enough|if .{0,45}enough)\b", re.I)
_UNCHANGED_CAPITAL = re.compile(r"\b(?:leave|keep|leaving|keeping)\b.{0,45}\b(?:capital|principal|original (?:savings|amount))\b|\b(?:capital|principal|original (?:savings|amount))\b.{0,35}\b(?:in place|intact|unchanged)\b", re.I)
_NO_USE = re.compile(r"\b(?:unused|untouched|stay(?:s)? (?:saved|put)|kept saved|reinvest\w*)\b|\b(?:" + _NOT_STARTED_TEXT + r"|" + _NOT_RECEIVED_TEXT + r"|" + _INABILITY_USE_TEXT + r"|won'?t|don'?t|do not|doesn'?t|does not|never|not|no longer)\b.{0,25}\b(?:use|used|using|pay|pays|paying|paid|fund|funds|funding|funded|finance|finances|financing|financed|spend|spending|spent|draw|drawing|withdraw\w*|cover|covers|covering|covered|meet|meets|meeting)\b", re.I)
_CAPITAL_USE = re.compile(r"\b(?:withdraw\w*|drawdowns?|draw(?:ing|n)? (?:down|from)|draw(?:ing)? (?:my|our) (?:own )?savings|draw(?:ing)? money out of|dip(?:ping)? into|tak(?:e|ing) chunks from|sales? of (?:shares|units)|draw(?:ing|n)? (?:my |our |the )?(?:own )?saved (?:capital|money|cash|balance)|(?:my |our |the )?(?:own )?savings? drawn|sell|selling|sold|sale proceeds|take (?:money|it) (?:out|out of|from)|use up|(?:use|using|spend|spending) (?:my |our |the |that |this )?(?:own )?(?:savings|saved money|saved balance|capital)|transfers? (?:my |our |the )?(?:own )?saved (?:money|balance))\b", re.I)
_PENSION_HOLDING = re.compile(r"\b(?:pension|penson|penssion|pention|pensoin) (?:pot|balance|fund|holdings|account|savings)\b", re.I)
_PENSION_PAYMENT = re.compile(r"\b(?:pension|penson|penssion|pention|pensoin) (?:payments?|income|instalments?|receipts)\b|\b(?:receive|receiving|get|getting|paid)\b.{0,25}\b(?:pension|penson|penssion|pention|pensoin)\b", re.I)

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
    r"use|using|spend|spending|withdraw|withdraws|withdrawing|draw|draws|drawing|drawn|take|takes|taking|go|goes|going|supply|supplies|supplying|buy|buys|buying|"
    r"meet|meets|meeting|live|living|send|sends|sending|give|gives|giving|"
    r"transfer|transfers|transferring|support|supports|supporting|provide|provides|providing|contribute|contributes|contributing)\b|"
    r"\b(?:is|are) (?:paid|covered|funded|financed)\b", re.I)
_NON_CURRENT_PROOF = re.compile(
    r"\b(?:" + _NOT_STARTED_TEXT + r"|" + _FUTURE_CONTRACTION_TEXT + r"|will|would|could|might|may|future|next (?:week|month|year|term)|used to|previously|"
    r"formerly|for later|later plan|later payment|later income|yesterday|last (?:week|month|year)|in the past|had|was|were|stopped|ended)\b", re.I)
_ONE_OFF_PROOF = re.compile(r"\b(?:one[- ]off|one[- ]time|occasional|isolated|single (?:gift|payment))\b", re.I)
_DENIED_FOCAL_PROOF = re.compile(r"^(?:with no|without|no)\b", re.I)
_LABELS = {
    "D5_SALARY": r"(?:salary|wages?|employment pay)",
    "D5_SELF_EMPLOYMENT": r"(?:self[- ]employed earnings|self[- ]employment income|freelance earnings)",
    "D5_PENSION_PAYMENTS": r"(?:pension|pension payments?|pension income|retirement payments?)",
    "D5_BENEFITS_SUPPORT": r"(?:benefits?|universal credit|regular support|regular family support)",
    "D5_WITHDRAWALS": r"(?:savings|withdrawals(?: from (?:my |our )?(?:own )?savings)?|savings withdrawals)",
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
        r"daughters?|sons?|nephews?|nieces?|cousins?|relative|aunt|uncle|neighbour|neighbor)|his|her|their)\b)|"
        r"\s+(?=(?:with no|without)\b)",
        _text(original)) if c.strip(" ,")]
    result = list(clauses)
    for left, right in zip(clauses, clauses[1:]):
        if re.match(r"^(?:it|this (?:income|payment|money)|that (?:income|payment|money))\b", right, re.I):
            result.append(left + ". " + right)
    return tuple(result)


# A bounded D5 fragment may omit "my" and the verb, but the whole unit
# must name a source and a current living-cost use. Arbitrary mentions do not
# become contributors. These finite premises only veto a supplied candidate.
_FRAGMENT_SOURCE = {
    "D5_SALARY": r"(?:salary|wages?|employment pay|pay cheque|pay packet)",
    "D5_SELF_EMPLOYMENT": r"(?:self[- ]employment income|self[- ]employed earnings|freelance earnings)",
    "D5_PENSION_PAYMENTS": r"(?:pension payments?|pension income|retirement payments?)",
    "D5_BENEFITS_SUPPORT": r"(?:benefits?|universal credit|regular (?:family )?support)",
    "D5_WITHDRAWALS": r"(?:withdrawals? from (?:my |our )?(?:own )?savings?|savings? withdrawals?|(?:my |our )?(?:own )?savings? drawn|saved capital (?:i|we) own drawn)",
    "D5_OTHER_REGULAR": r"(?:(?:rent|rental) income|rent receipts?|dividends?|interest income|royalties)",
}
_FRAGMENT_USE = {
    category: re.compile(r"^(?:(?:my|our|own) )?" + source
        + r"(?: (?:coming in|received|used|spent|drawn|currently|now))*"
        + r" (?:for|on|towards?) (?:(?:my|our|the) )?(?:other )?" + _COST_WORDS
        + r"(?: (?:currently|now|at present))?[.!]?$", re.I)
    for category, source in _FRAGMENT_SOURCE.items()
}
_DIRECT_ABSENCE = re.compile(
    r"\b(?:nothing|no money|no source) (?:regular(?:ly)?|recurring|ongoing) "
    r"(?:is )?(?:funding|paying|covering|providing money for) (?:my |our |the )?"
    + _COST_WORDS + r"\b", re.I)
_DENIAL = re.compile(r"\b(?:" + _NOT_STARTED_TEXT + r"|" + _NOT_RECEIVED_TEXT + r"|" + _INABILITY_USE_TEXT + r"|won'?t|neither|no|none|nothing|without|not|never|don'?t|do not|doesn'?t|does not|"
    r"isn't|is not|aren't|are not|no longer|unused|untouched|stay saved|stays saved|"
    r"kept saved|stay(?:s)? put|reinvest\w*)\b", re.I)
_NON_ABSENCE_NEGATION = re.compile(r"\b(?:not|no) (?:only|alone|just|entirely|merely)\b|"
    r"\bnot (?:the other way round|dividends? alone)\b", re.I)
_SOURCE_COORDINATION = re.compile(
    r"\s+(?:and|n|&)\s+(?=(?:(?:i|we|he|she|they|it|my|our|his|her|their)\b|"
    r"(?:next (?:week|month|year)|in the future|draw|withdraw|spend|use|receive|rent income|rental income|wages|salary|benefits|"
    r"pension payments|interest income|dividends|savings withdrawals)\b))", re.I)
_NOMINAL_SOURCE = re.compile(r"^(?:(?:my|our|own) )?(?:"
    + "|".join(_FRAGMENT_SOURCE.values()) + r")[.!]?$", re.I)
_RECEIVED_SUPPORT = re.compile(
    r"\b(?:regular|recurring|ongoing) money from (?:my |our )?"
    r"(?:friend|family|parent|mother|father|mum|dad|sister|brother|partner|relative)s?"
    r" to (?:me|us)\b", re.I)
_UNACCOUNTED_SOURCE_USE = re.compile(
    r"\b(?:another|unknown|unnamed|unidentified|some other) (?:regular |current )?"
    r"(?:source|income|money|funding)\b.{0,55}\b(?:pay|pays|cover|covers|fund|funds|use|uses)\b"
    r".{0,55}\b" + _COST_WORDS + r"\b", re.I)
_CURRENT_SCOPE = re.compile(r"\b(?:currently|now|at present|atm|today)\b", re.I)
_GENERIC_RECEIVED = re.compile(r"\b(?:receive|receiving|get|getting) (?:it|them|money|support)\b", re.I)


_EXCLUDED_SOURCE_PREFIX = re.compile(r"(?:rather than|instead of|as opposed to|not from)\s+(?:(?:my|our|any|the)\s+)?$", re.I)
_ABSENCE_ITEM = r"(?:(?:continuing|regular|current|ongoing|recurring|scheduled) )*(?:financial support|support|regular income|income|savings withdrawals|capital draws|receipts)"
_ABSENCE_LIST = re.compile(r"^(?:i|we) (?:have|get|receive) no " + _ABSENCE_ITEM
    + r"(?:\s*(?:,|or|and)\s*" + _ABSENCE_ITEM + r")+"
    + r"(?: for (?:my |our )?(?:everyday living|living costs|living expenses))?[.!]?$", re.I)


def _source_is_explicitly_excluded(unit: str, category: str) -> bool:
    mentions = list(PATTERNS[category].finditer(unit))
    return bool(mentions) and all(_EXCLUDED_SOURCE_PREFIX.search(unit[max(0,m.start()-50):m.start()])
                                  for m in mentions)


_NOUN_FUNDING = re.compile(
    r"\b(?:is|are) (?:what (?:i|we) (?:use|spend)|(?:my|our) (?:sole |only |current |regular )*"
    r"(?:living[- ]cost money|money for living|money for (?:normal )?living costs)|"
    r"(?:my |our )?(?:sole |only |current |regular )*money for (?:(?:normal )?living costs|(?:daily|everyday|ordinary|normal) living)|(?:my|our) living money)\b", re.I)
_ASSET_HISTORY = re.compile(r"\b(?:money|funds|cash|capital) (?:i|we) (?:previously|formerly) saved\b", re.I)
_INCOME_ON_ASSET = re.compile(
    r"\b(?:dividends?|interest(?: income)?) (?:received |earned )?(?:from|on) (?:my |our |the )?"
    r"(?:retained |existing |owned )?(?:shares|savings|capital|investments|funds)\b", re.I)


def _asset_context(unit: str) -> str:
    # Past saving describes asset history, not necessarily past withdrawal.
    # The exact original evidence is unchanged and still checked separately.
    return _ASSET_HISTORY.sub(lambda match: re.sub(r"\b(?:previously|formerly)\b", "", match.group()), unit)


def _has_funding_relation(unit: str) -> bool:
    return bool(_COST_USE.search(unit) or _NOUN_FUNDING.search(unit))


_LINKED_USE = re.compile(
    r"^(?:it|they|both|that(?:'s| is)?|this (?:income|payment|money)|"
    r"that (?:income|payment|money)|those (?:income )?receipts|"
    r"the (?:current )?(?:profit|earnings)|(?:both|these) pension receipts|"
    r"(?:i|we|im|i'm) (?:am |are )?(?:receive and )?"
    r"(?:use|using|spend|spending|do not draw|don't draw|do not use|don't use) "
    r"(?:it|them|both|the actual payments|that support|this support)|"
    r"(?:use|using|spend|spending) (?:that|this) capital)\b", re.I)
_PERSONAL_SOURCE = re.compile(r"\b(?:i|we|my|our|me|us)\b", re.I)
_OWN_WORK = re.compile(
    r"\b(?:my|our) own (?:[a-z-]+ ){1,3}(?:sales|jobs)\b", re.I)
_NOT_EMPLOYED_WORK = re.compile(
    r"\b(?:neither|none|not) (?:is |are )?(?:employed|employee|employment) (?:work|pay|income|jobs)\b", re.I)
_DONOR_ANAPHORA = re.compile(
    r"^(?:he|she|they) (?:regularly|monthly|weekly) (?:sends?|gives?|transfers?) (?:me|us) money\b", re.I)


def _contributor_units(original: str) -> tuple[str, ...]:
    """Scope finite funding units without detaching context from its source.

    A comma only separates two independently named sources. Expense context,
    enumerated work costs and shared income predicates stay attached. A new
    first-person clause can still introduce a different source; short adjacent
    own-business/receipt anaphora may share a directly asserted current use.
    """
    clauses = _financial_focal_clauses(original)
    result = []
    for clause in clauses:
        comma_units = re.split(r",\s*", clause)
        scoped_commas = []
        pending = ""
        for comma in comma_units:
            if not pending:
                pending = comma
                continue
            left = {k for k,p in PATTERNS.items() if p.search(pending)}
            right = {k for k,p in PATTERNS.items() if p.search(comma)}
            # A negated/future unit has its own scope. It cannot erase a
            # preceding current-source predicate or modifier-prefix context.
            absence_prefix = bool(re.search(r"\b(?:no|without) (?:income|funding|support|receipts)\b", pending, re.I))
            new_scoped_source = bool(right and left and not absence_prefix and not _ABSENCE_LIST.fullmatch(pending + ", " + comma) and (right - left or _relation_denied(comma)
                or _NON_CURRENT_PROOF.search(comma)))
            if new_scoped_source:
                scoped_commas.append(pending)
                pending = comma
            else:
                pending = pending + ", " + comma
        if pending:
            scoped_commas.append(pending)
        for comma in scoped_commas:
            units = _SOURCE_COORDINATION.split(comma)
            pending = ""
            for unit in units:
                unit = unit.strip(" ,")
                if not unit:
                    continue
                if pending:
                    left = {k for k,p in PATTERNS.items() if p.search(pending)}
                    right = {k for k,p in PATTERNS.items() if p.search(unit)}
                    if (_NOMINAL_SOURCE.fullmatch(pending) or
                            re.search(r"\b(?:receive|get)$", pending, re.I) and re.match(r"^(?:use|spend)\b", unit, re.I) or
                            left and not _has_funding_relation(pending)
                            and (not right or right <= left)
                            and not _NON_CURRENT_PROOF.search(unit) and not _relation_denied(unit)):
                        pending = pending + " and " + unit
                        continue
                    result.append(pending)
                pending = unit
            if pending:
                result.append(pending)
    # Explicit same-source receipt/profit anaphora can attach immediately
    # following cost use. No generic owner, time or pronoun inference occurs.
    linked_left = set()
    original_units = tuple(result)
    for index, (left, right) in enumerate(zip(original_units, original_units[1:])):
        if (not _has_funding_relation(left) and any(p.search(left) for p in PATTERNS.values())
                and _LINKED_USE.search(right)
                and (_has_funding_relation(right) or _NO_USE.search(right))):
            result.append(left + ". " + right)
            linked_left.add(index)
        elif (_OTHER_SUBJECT.search(left) and PATTERNS["D5_BENEFITS_SUPPORT"].search(left)
                and _DONOR_ANAPHORA.search(right) and _has_funding_relation(right)
                and not _NON_CURRENT_PROOF.search(right) and not _relation_denied(right)):
            # Only the immediately named family donor can anchor this pronoun.
            # Generic transfers from an employer are not support evidence.
            result.append(left + ", " + right)
            linked_left.add(index)
    # The directly linked unit carries exactly the source, owner and
    # current/inactive predicate; do not also veto its now-contextless noun.
    result = [unit for index, unit in enumerate(result) if index not in linked_left]
    # The question can contain multiple own work receipts in one category.
    # This bounded proof needs both an own-work earnings predicate and explicit
    # non-employment context. Owning an asset or merely saying "paid jobs"
    # does not supply this premise.
    if _NOT_EMPLOYED_WORK.search(original) and _OWN_WORK.search(original):
        for unit in tuple(result):
            if _OWN_WORK.search(unit) and _has_funding_relation(unit):
                result.append(unit + ". " + _NOT_EMPLOYED_WORK.search(original).group())
    return tuple(result)


def _relation_denied(unit: str) -> bool:
    # Narrow exceptions are denial of exclusivity/direction, not no-use.
    scoped = _NON_ABSENCE_NEGATION.sub("", unit)
    return bool(_DENIAL.search(scoped))


def current_no_regular_premise(original: str) -> bool:
    """Explicit present customer absence only; not a replacement answer."""
    text = _text(original).strip()
    for clause in _financial_focal_clauses(text):
        if not (PATTERNS["D5_NO_REGULAR"].search(clause) or _DIRECT_ABSENCE.search(clause)
                or _ABSENCE_LIST.fullmatch(clause.strip()) and re.search(r"\bregular income\b", clause, re.I)):
            continue
        if (_NON_CURRENT_PROOF.search(clause) or _OTHER.search(clause)
                or _OTHER_SUBJECT.search(clause) or _UNCERTAIN.search(clause)
                or re.search(r"\b(?:except|apart from|other than|unless|example|article|"
                    r"hypothetical|suppose|not true|not the case)\b", clause, re.I)):
            continue
        return True
    return False


def received_support_premise(original: str) -> bool:
    """Only an explicit received/current support relationship can pass."""
    units = _contributor_units(original)
    for index, unit in enumerate(units):
        if (_NON_CURRENT_PROOF.search(unit) or _UNCERTAIN.search(unit)
                or _ONE_OFF_PROOF.search(unit) or _relation_denied(unit)):
            continue
        if not PATTERNS["D5_BENEFITS_SUPPORT"].search(unit):
            continue
        recipient = bool(_RECIPIENT.search(unit) or _RECEIVED_SUPPORT.search(unit)
            or _PERSONAL_SOURCE.search(unit) and not (_OTHER.search(unit) or _OTHER_SUBJECT.search(unit)))
        if _has_funding_relation(unit) and recipient:
            return True
        if recipient and index + 1 < len(units):
            following = units[index + 1]
            if (re.match(r"^(?:i|we) (?:receive and )?(?:use|spend) (?:it|them|both|that support|this support)\b", following, re.I)
                    and _COST_USE.search(following) and not _NON_CURRENT_PROOF.search(following)
                    and not _relation_denied(following) and not _UNCERTAIN.search(following)):
                return True
    return False


def _unit_positive_sources(unit: str) -> frozenset[str]:
    # A final explicit denial of employed work limits own sales/jobs; it
    # does not negate those receipts. Keep other negation untouched.
    own_work_context = bool(_OWN_WORK.search(unit) and _NOT_EMPLOYED_WORK.search(unit))
    scoped_unit = _NOT_EMPLOYED_WORK.sub("", unit) if own_work_context else unit
    if (_NON_CURRENT_PROOF.search(_asset_context(scoped_unit)) or _relation_denied(scoped_unit)
            or _INABILITY_PAY.search(unit) or _UNCERTAIN.search(unit) or _ONE_OFF_PROOF.search(unit)):
        return frozenset()
    fragments = {key for key, pattern in _FRAGMENT_USE.items() if pattern.fullmatch(unit)}
    if fragments:
        return frozenset(fragments)
    if _LABEL_LIST.fullmatch(unit):
        return frozenset(k for k, pattern in _LABELS.items() if re.search(pattern, unit, re.I))
    if not _has_funding_relation(unit):
        return frozenset()
    # A source may have a first-person recipient embedded after the noun:
    # "draw saved capital I own for food" is sufficient, but a held balance is not.
    positive = set()
    if own_work_context and _PERSONAL_SOURCE.search(unit):
        positive.add("D5_SELF_EMPLOYMENT")
    for category, pattern in PATTERNS.items():
        if category == "D5_NO_REGULAR" or not pattern.search(scoped_unit) or _source_is_explicitly_excluded(scoped_unit, category):
            continue
        if category == "D5_BENEFITS_SUPPORT":
            if (_OTHER.search(unit) or _OTHER_SUBJECT.search(unit)) and not _RECIPIENT.search(unit):
                continue
        elif (_OTHER.search(unit) or _OTHER_SUBJECT.search(unit)):
            continue
        if category == "D5_WITHDRAWALS" and not _CAPITAL_USE.search(unit):
            continue
        if category == "D5_PENSION_PAYMENTS" and _PENSION_HOLDING.search(unit) and not _PENSION_PAYMENT.search(unit):
            continue
        positive.add(category)
    return frozenset(positive)


def positive_current_source_categories(original: str) -> frozenset[str]:
    """Finite necessary current-use premises; never propose an option."""
    categories = {category for unit in _contributor_units(_text(original).strip())
                  for category in _unit_positive_sources(unit)}
    if received_support_premise(original):
        categories.add("D5_BENEFITS_SUPPORT")
    return frozenset(categories)



_RECEIPT_DENIAL = re.compile(
    r"\b(?:i|we) (?:(?:don'?t|do not|never) (?:receive|get|have)|"
    + _NOT_RECEIVED_TEXT + r")\b|"
    r"\b(?:unused|untouched|stay saved|stays saved|kept saved)\b", re.I)


_ALL_LIVING_COSTS = re.compile(
    r"\b(?:any |my |our |the |all (?:my |our )?)?"
    r"(?:living|daily|everyday|day[- ]to[- ]day) (?:costs|expenses)\b", re.I)


def current_source_relation_conflicts(original: str) -> tuple[str, ...]:
    """Veto unresolved same-source current assertions and explicit denials.

    A claim that received pension pays costs cannot coexist with 'I do not
    receive any pension'. Distinct spending areas may have different payers;
    a denial scoped only to rent does not erase a source funding food. No
    replacement answer or general-purpose pronoun inference is performed.
    """
    active = positive_current_source_categories(original)
    conflicts = set()
    for unit in _contributor_units(_text(original)):
        if (_NON_CURRENT_PROOF.search(unit) or _OTHER.search(unit)
                or _OTHER_SUBJECT.search(unit)):
            continue
        if not _relation_denied(unit) or not (_RECEIPT_DENIAL.search(unit)
                or _NO_USE.search(unit) and _ALL_LIVING_COSTS.search(unit)):
            continue
        for category, pattern in PATTERNS.items():
            if category in active and pattern.search(unit):
                conflicts.add(category)
    return tuple("current_source_assertion_and_receipt_denial_conflict:" + category
                 for category in sorted(conflicts))

def candidate_current_premise_violations(original: str, option: str) -> tuple[str, ...]:
    """Require a necessary positive premise for an existing single candidate.

    A named-but-denied source cannot pass merely because no different current
    source was detected. An unqualified exact label inherits this question's
    context; narrative negation, future, holdings or other-owner claims do not.
    """
    if option in {"D5_NO_REGULAR", "D5_MULTIPLE"}:
        return ()
    text = _text(original).strip()
    if text.lower().rstrip(".!?") == option.lower():
        return ()
    if option in positive_current_source_categories(text):
        return ()
    if option == "D5_BENEFITS_SUPPORT" and received_support_premise(text):
        return ()
    # Short recognised support abbreviations cannot imply another category.
    if option == "D5_BENEFITS_SUPPORT" and re.fullmatch(r"(?:pip|dla|uc|esa|jsa|maintenance|alimony)[.!]?", text, re.I):
        return ()
    return ("candidate_current_customer_use_not_established",)

def contributor_completeness_violations(original: str, option: str) -> tuple[str, ...]:
    """Veto a single-source proposal with an unaccounted relevant source.

    Every recognised residual source must be explicitly inactive, future,
    another person's, or positively supported in the same proposed category.
    This cannot create an option. An unresolved contributor is model/business
    uncertainty, not user Undecided. MULTIPLE still needs two current sources.
    """
    if option == "D5_MULTIPLE":
        return ()
    active = positive_current_source_categories(original)
    issues = []
    if option != "D5_NO_REGULAR" and active - {option}:
        issues.append("candidate_omits_current_contributor")
    if option == "D5_NO_REGULAR" and active:
        issues.append("absence_conflicts_with_current_contributor")
    for unit in _contributor_units(_text(original)):
        mentioned = {key for key, pattern in PATTERNS.items()
                     if key != "D5_NO_REGULAR" and pattern.search(unit)}
        if not mentioned:
            if (_UNACCOUNTED_SOURCE_USE.search(unit) and not _NON_CURRENT_PROOF.search(unit)
                    and not _relation_denied(unit)):
                issues.append("unnamed_current_contributor_not_accounted_for")
            continue
        unit_active = _unit_positive_sources(unit)
        inactive = (_NON_CURRENT_PROOF.search(_asset_context(unit)) or _relation_denied(unit)
            or _ONE_OFF_PROOF.search(unit) or _PENSION_HOLDING.search(unit))
        if inactive:
            continue
        for category in mentioned - unit_active:
            if _source_is_explicitly_excluded(unit, category):
                continue
            if category == "D5_BENEFITS_SUPPORT" and unit_active == {"D5_BENEFITS_SUPPORT"}:
                continue
            # A donor's salary/pension/capital finances support, not customer's
            # salary/pension/withdrawals. It remains represented as other-owner.
            if category != "D5_BENEFITS_SUPPORT" and (_OTHER.search(unit) or _OTHER_SUBJECT.search(unit)):
                continue
            if category == "D5_WITHDRAWALS" and (_UNCHANGED_CAPITAL.search(unit)
                    or _INCOME_ON_ASSET.search(unit) and not _CAPITAL_USE.search(unit)):
                continue
            # Donor-only income is not evidence of received aid either.
            if category == "D5_BENEFITS_SUPPORT" and _OTHER_SUBJECT.search(unit) and not _RECIPIENT.search(unit):
                continue
            if category != option or option == "D5_NO_REGULAR":
                issues.append("residual_source_contribution_not_established")
    return tuple(dict.fromkeys(issues))


def grounding_violations(original: str, facts: Sequence[Mapping[str, str]], option: str) -> tuple[str, ...]:
    """Return veto reasons only. Never return an option, bucket or repaired fact."""
    text = _text(original)
    issues = list(contributor_completeness_violations(original, option))
    issues.extend(candidate_current_premise_violations(original, option))
    issues.extend(current_source_relation_conflicts(original))
    if option == "D5_NO_REGULAR" and not current_no_regular_premise(original):
        issues.append("current_no_regular_relation_not_supported")
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
        terse = (bool(_FRAGMENT_USE.get(category) and _FRAGMENT_USE[category].fullmatch(normalized.strip()))
            or normalized.strip() == text.strip() and bool(re.fullmatch(
            r"(?:my |our )?(?:salary|wages|pension payments|benefits|universal credit|regular support|withdrawals(?: from savings)?|rental income|rent income|interest income|dividends)[.!]?", normalized.strip(), re.I)))
        if not terse and not _has_funding_relation(normalized):
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
