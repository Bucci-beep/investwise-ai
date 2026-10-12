"""Veto-only business validation of an ML-proposed D5 candidate.

This module never chooses funding categories or decides active contributors.
The learned model must supply an allowed candidate first. These small guards
reject explicit uncertainty, a missing/wrong question premise, or an absent
necessary lexical premise; success returns exactly the supplied candidate.
Pre-model urgent, care and control handling stays in the conversation pipeline.
"""
from __future__ import annotations

import re

from ai.bounded_answers import BoundedAnswer, supported_detail
from ai.question_spec import BusinessSpec


_FUNDING_PROMPT = "Which source or sources actually provide money for your regular living costs now?"
_UNCERTAINTY = re.compile(
    r"\b(?:not sure|unsure|uncertain|don't know|do not know|can't tell|cannot tell|"
    r"can't identify|cannot identify|can't establish|cannot establish|can't determine|"
    r"cannot determine|can't remember|cannot remember|haven't worked out|have not worked out|"
    r"haven't established|have not established|haven't resolved|have not resolved|"
    r"can't choose|cannot choose|can't decide|cannot decide|undecided|torn between)\b"
)
_CURRENT_SUBJECT = re.compile(
    r"\b(?:living (?:costs|expenses)|everyday (?:costs|expenses|spending)|"
    r"day[- ]to[- ]day|household (?:bills|costs|expenses)|usual (?:bills|costs|expenses)|"
    r"regular (?:source|sources)|current (?:source|sources|funding)|"
    r"which (?:one|source|sources|account)|actually (?:use|spend|pay|fund)|"
    r"support(?:s|ing)? me|(?:fund|funds|funding|pay|pays|paying) (?:my|the) (?:bills|costs|expenses))\b"
)
_NON_CURRENT = re.compile(
    r"\b(?:used to|previously|in the past|last (?:week|month|year|winter|spring|summer|autumn)|"
    r"will (?:receive|get|earn|start|pay|fund|be|withdraw|draw|use|take)|i'll (?:receive|get|earn|start|withdraw|draw|use|take)|won't start|will not start|"
    r"(?:doesn't|does not|hasn't|has not) (?:start|started)|"
    r"future (?:income|salary|wages|payments)|next (?:week|month|year|winter|spring|summer|autumn)|"
    r"plan to (?:earn|receive|get|draw|use|withdraw)|hope to (?:get|receive|earn)|"
    r"might (?:get|receive|earn|draw|use|withdraw)|"
    r"in the future|later|has ended|has stopped|stopped (?:months|years) ago)\b"
)
_FUTURE = re.compile(
    r"\b(?:will (?:receive|get|earn|start|pay|fund|be|withdraw|draw|use|take)|i'll (?:receive|get|earn|start|withdraw|draw|use|take)|won't start|will not start|"
    r"future (?:income|salary|wages|payments)|next (?:week|month|year|winter|spring|summer|autumn)|"
    r"in the future|plan to (?:earn|receive|get|draw|use|withdraw)|hope to (?:get|receive|earn)|"
    r"might (?:get|receive|earn|draw|use|withdraw)|not started yet)\b"
)
_FUNDING_ASSERTION = re.compile(
    r"\b(?:live|living) on\b|"
    r"\b(?:pay|pays|paying|paid|cover|covers|covering|fund|funds|funding|finance|finances|financed)\b"
    r".{0,95}\b(?:living|everyday|day[- ]to[- ]day|bills|costs|expenses|food|groceries|heating|transport)\b|"
    r"\b(?:living costs|living expenses|everyday costs|household bills|usual expenses)\b"
    r".{0,70}\b(?:paid|funded|financed|covered)\b|"
    r"\b(?:use|using|spend|spending)\b.{0,95}\b(?:living costs|living expenses|bills|food|groceries)\b|"
    r"\b(?:go|goes|going) (?:on|towards?|to) (?:my |our |the )?"
    r"(?:living costs|living expenses|bills|food|groceries|rent|heating)\b|"
    r"\b(?:current|regular) (?:funding|source)\b"
)
_POSSESSION_ONLY = re.compile(r"^(?:i (?:have|own)|i've got|i have got|my (?:account|pension) balance)\b")
_HELD_PENSION = re.compile(
    r"\bpension (?:balance|pot|holdings|savings|fund)\b|"
    r"\b(?:held|holding|inside|in) (?:my|a|the) pension\b"
)
_PENSION_PAYMENT_ASSERTION = re.compile(
    r"\b(?:receive|receiving|get|getting|paid|pays)\b.{0,45}\bpension (?:payments|income)\b|"
    r"\bpension\b.{0,40}\b(?:pays|paying|paid out|paying out)\b"
)
_SAME_MONEY = re.compile(
    r"\b(?:this (?:exact |same )?pot|this (?:named )?reserve|same reserve|"
    r"same money|money we are discussing|money we're discussing)\b"
)
_DEPENDENCY_ACTION = re.compile(
    r"\b(?:withdraw\w*|draw(?:ing)?|take|taking|use|using|spend|spending|live|living)\b"
    r".{0,85}\b(?:this (?:exact |same )?pot|this (?:named )?reserve|same reserve|"
    r"same money|money we are discussing|money we're discussing)\b|"
    r"\b(?:this (?:exact |same )?pot|same reserve|same money)\b"
    r".{0,60}\b(?:pays|funds|covers)\b.{0,50}\b(?:costs|bills|expenses|food|groceries)\b"
)
_EXCLUSIVE_CLAIM = re.compile(
    r"(?P<claim>[^.;!?]+?)\b(?:is|are) (?:my|the) (?:only|sole) "
    r"(?:(?:current|regular|income) )*(?:source|funding)\b"
)
_VAGUE_DETAIL_ONLY = re.compile(
    r"^(?:(?:i (?:use|have|get|receive)|it's|it is|from) )?"
    r"(?:another|an? other|some|an? unknown|an? unnamed) (?:regular )?(?:source|income|money)"
    r"(?: (?:for|that pays|paying|to pay) (?:my )?(?:living costs|living expenses|bills))?[.!]?$")
_UNNAMED_MIX_ONLY = re.compile(
    r"^(?:multiple(?: sources)?|more than one (?:listed )?(?:source(?: category)?|income)|"
    r"a (?:mix|mixture|combination) of (?:sources|income)|several sources)[.!]?$"
)
_UNNAMED_CURRENT_DETAIL = re.compile(
    r"\b(?:another|some|unknown|unnamed|unidentified) (?:other |regular )*(?:source|income|money)\b"
    r".{0,55}\b(?:pays|paying|funds|funding|covers|covering)\b.{0,55}\b(?:costs|bills|expenses|food|groceries)\b|"
    r"\b(?:use|using|spend|spending)\b.{0,55}"
    r"\b(?:another|some|unknown|unnamed|unidentified) (?:other |regular )*(?:source|income|money)\b"
    r".{0,55}\b(?:costs|bills|expenses|food|groceries)\b"
)
_DATE_TIMING = re.compile(
    r"\b(?:starts?|begins?|starting|beginning)\b.{0,35}\b(?:next|in|on|from|until)\b"
    r".{0,25}\b(?:spring|summer|autumn|winter|week|month|year|january|february|march|"
    r"april|may|june|july|august|september|october|november|december|20\d{2})\b"
)
_OTHER_OWNER = re.compile(r"\b(?:my (?:brother|sister|neighbour|neighbor|partner|mother|father|friend|relative)|someone else)\b")
_OWN_COSTS = re.compile(r"\b(?:(?:my|our) (?:(?:joint|shared|living|everyday|household) )*(?:costs|bills|expenses|food|groceries|rent)|support(?:s|ing)? (?:me|us))\b")
_OTHER_SUBJECT = re.compile(
    r"^(?:my (?:brother|sister|neighbour|neighbor|partner|mother|father|friend|relative|dad|mum|parents?)|he|she|they)\b"
)
_THIRD_PARTY_FUNDING_CONTEXT = re.compile(
    r"\b(?:(?:my|our|his|her|their|the) (?:wife|husband|spouse|partner|"
    r"parents?|mother|father|mum|mom|dad|brothers?|sisters?|siblings?|"
    r"friends?|relatives?|cousins?|aunt|uncle|grandparents?|grandmother|grandfather|"
    r"colleagues?|coworkers?|neighbours?|neighbors?|flatmates?|housemates?|roommates?)|"
    r"someone else|another person|other people|he|she|they|his|her|their|your|"
    r"dad|mum|mom|mother|father)\b"
)
_FUNDING_TOPIC = re.compile(
    r"\b(?:money|income|funding|source|pay|pays|paying|paid|uses?|using|"
    r"earns?|receives?|support|help|allowance|bills|costs|expenses)\b"
)
_OTHER_PERSON_COSTS = re.compile(
    r"(?:\b(?:his|her|their|your)|"
    + _THIRD_PARTY_FUNDING_CONTEXT.pattern + r"(?:'s|')) "
    r"(?:(?:own|regular|living|everyday|household) )*"
    r"(?:costs|bills|expenses|food|groceries|rent|heating|transport)\b"
)
_CUSTOMER_RECEIPT = re.compile(
    r"\b(?:i|we) (?:(?:currently|regularly|now) )?"
    r"(?:receive|receiving|get|getting|earn|earning|use|using|spend|spending|live|living)\b|"
    r"\b(?:send|sends|sending|give|gives|giving|transfer|transfers|transferring)\b"
    r".{0,45}\b(?:me|us)\b|"
    r"\b(?:come|comes|coming) to (?:me|us)\b"
)
_CONTEXTUAL_SUPPORT_FRAGMENT = re.compile(
    r"\b(?:help|support|assistance|money)(?: payments?)? from\b|"
    r"\bfrom\b.{0,65}\b(?:help|support|assistance|money)\b"
)
_NEGATED_FUNDING = re.compile(
    r"\b(?:don't|do not|doesn't|does not|never|not|isn't|is not|aren't|are not|no|none|nothing)\b"
)
_WITHDRAWAL_RESOURCE = r"(?:savings?|investments?|reserves?|pot|saved cash|shares|funds)"
_WITHDRAWAL_RESOURCE_WORD = re.compile(r"\b" + _WITHDRAWAL_RESOURCE + r"\b")
_OWN_WITHDRAWAL_DENIAL = re.compile(
    r"\b(?:i|we) (?:don't|do not|never|(?:am|are) not|no longer) "
    r"(?:(?:currently|regularly|now) )?"
    r"(?:withdraw(?:ing)?|draw(?:ing)?(?: down)?|take|taking|use|using|spend|spending) "
    r"(?:money (?:out of|from) |(?:out of|from) )?(?:my|our) (?:own )?" + _WITHDRAWAL_RESOURCE + r"\b|"
    r"\bno (?:current |regular )?withdrawals? from (?:my|our) (?:own )?" + _WITHDRAWAL_RESOURCE + r"\b|"
    r"\b(?:my|our) (?:own )?" + _WITHDRAWAL_RESOURCE + r" "
    r"(?:stay|stays|remain|remains|are|is) (?:entirely |completely )?(?:untouched|unused)\b"
)
_OWN_WITHDRAWAL_ACTION = re.compile(
    r"\b(?:i|we) (?:(?:currently|regularly|now) )?"
    r"(?:withdraw(?:ing)?|draw(?:ing)?(?: down)?|take|taking|use|using|spend|spending|sell|selling) "
    r"(?:money (?:out of|from) |(?:out of|from) )?(?:my|our) (?:own )?"
    + _WITHDRAWAL_RESOURCE + r"\b|"
    r"\b(?:i|we) (?:(?:currently|regularly|now) )?"
    r"(?:withdraw(?:ing)?|draw(?:ing)?(?: down)?|spend|spending) "
    + _WITHDRAWAL_RESOURCE + r"\b|"
    r"\b(?:my|our) (?:own )?" + _WITHDRAWAL_RESOURCE + r"\b.{0,45}"
    r"\b(?:withdrawals?|drawdowns?|pay|pays|cover|covers|fund|funds)\b"
)
_OTHER_WITHDRAWAL_RESOURCE = re.compile(
    r"\b(?:his|her|their) (?:own )?" + _WITHDRAWAL_RESOURCE + r"\b|"
    r"\bmy (?:brother|sister|partner|mother|father|friend|relative|dad|mum|parents?)'s "
    r"(?:own )?" + _WITHDRAWAL_RESOURCE + r"\b"
)
_WORKPLACE_PAY_MENTION = re.compile(r"\b(?:café|cafe|shop|restaurant|bar|agency|retail) pay\b")
_NO_REGULAR_ASSERTION = re.compile(r"\bno (?:current |regular |recurring )*(?:regular source|recurring source|regular income|regular funding)\b")
_ONE_OFF_RESOURCE = re.compile(r"\b(?:one[- ]off|one[- ]time|non[- ]recurring|occasional|isolated|single (?:cash )?(?:gift|payment|present|refund))\b")
_HYPOTHETICAL = re.compile(r"\b(?:if|hypothetical|hypothetically|imagine|suppose|would|could|might|may)\b")
_SUPPORT_PREMISE = re.compile(
    # Supplemental *mentions*, not evidence that support contributes. The
    # source analyzer already covers benefits, family and regular/financial
    # support; these common transfer/support labels must not be false vetoes.
    r"\b(?:maintenance|alimony|welfare|social security|tax credits?|income support|"
    r"child support|(?:support|assistance) payments?|jobseeker'?s? allowance|employment and support allowance|"
    r"personal independence payments?|attendance allowance|carer'?s? allowance|"
    r"disability living allowance|grant (?:aid|support)|"
    r"godparents?|godfather|godmother|guardian|benefactor|donor|folks)\b|"
    r"\b(?:my|the|an?) ex\b|"
    r"\b(?:regular|recurring|ongoing|continuing|weekly|fortnightly|monthly|financial) "
    r"(?:\w+ ){0,2}(?:help|support|assistance|allowance|aid)\b|"
    r"\b(?:from|by) (?:someone(?: else)?|somebody|another person|other people|others)\b|"
    r"\b(?:send|sends|sent|sending|give|gives|gave|giving|gift|gifts|gifted)\b"
    r".{0,35}\b(?:me|us)\b|"
    r"^(?:(?:only|just) )?(?:support|help|assistance|maintenance|alimony|uc|esa|jsa|pip|dla)[.!]?$"
)


def _normalise(text: str) -> str:
    return " ".join(text.lower().replace("’", "'").replace("—", "-").replace("–", "-").split())


def _current_answer(raw: str) -> str:
    """An explicit correction supersedes earlier claims, without selecting an option."""
    parts = re.split(
        r"(?:\bcorrection:\s*|\bno,? i mean\b|\bsorry,? i meant\b|(?:^|[.;]\s*)actually\b)",
        raw, flags=re.I,
    )
    return parts[-1].strip() if len(parts) > 1 and parts[-1].strip() else raw.strip()


def _clauses(text: str) -> list[str]:
    # Keep coordinated noun subjects attached to their shared predicate:
    # 'my wages and the monthly help both go on bills'. Split a new personal
    # subject, or a possessive subject with its own modal/negative predicate.
    # Clause boundaries scope vetoes only; they never extract categories.
    return [part.strip(" ,") for part in re.split(
        r"[.;!?]|\b(?:but|however|whereas)\b|"
        r"\band\s+(?=(?:i|we|he|she|it|they)\b|"
        r"(?:my|our|the)\s+[^,.;!?]{1,60}?\s+"
        r"(?:will|won't|would|could|might|may|doesn't|does not|isn't|is not|aren't|are not|never|"
        r"used to|previously|formerly|was|were|has ended|has stopped)\b)|"
        r",\s+(?=(?:i|my|our|the|he|she|it|they)\b)",
        text,
    ) if part.strip(" ,")]


def _has_current_funding_assertion(text: str) -> bool:
    for clause in _clauses(_without_quoted_material(text)):
        if _NON_CURRENT.search(clause) or _HYPOTHETICAL.search(clause):
            continue
        if _OTHER_SUBJECT.search(clause) and not _OWN_COSTS.search(clause):
            continue
        for match in _FUNDING_ASSERTION.finditer(clause):
            # A negated or hypothetical use cannot justify a current premise
            # merely because another part of the reply mentions future money.
            prefix = clause[max(0, match.start() - 45):match.start()]
            if not _NEGATED_FUNDING.search(prefix + match.group()):
                return True
    return False


def _other_person_only_funding_premise(text: str) -> bool:
    """Veto an explicit other-person account with no customer funding premise.

    This question-scope check does not identify a contributing category. Any
    separate current customer premise prevents the veto; the supplied model
    still decides its category. Unowned terse labels keep the question context.
    """
    from ai.d5_text_classifier import source_mentions

    other_person_context = False
    customer_premise = False
    for clause in _clauses(_without_quoted_material(text)):
        mentions = source_mentions(clause)
        other_person_spans = [match.span() for match in _THIRD_PARTY_FUNDING_CONTEXT.finditer(clause)]
        other_person = bool(other_person_spans)
        # Family words themselves are support *mentions*. They are not enough
        # to establish a funding topic or the customer's receipt of money.
        source_spans = [(mention.start, mention.end) for mention in mentions
                        if not any(mention.start < end and start < mention.end
                                   for start, end in other_person_spans)]
        source_spans.extend(match.span() for match in _WORKPLACE_PAY_MENTION.finditer(clause))
        if other_person and (source_spans or _FUNDING_TOPIC.search(clause)):
            other_person_context = True
        if _NON_CURRENT.search(clause) or _DATE_TIMING.search(clause) or _HYPOTHETICAL.search(clause):
            continue
        if not other_person and _NO_REGULAR_ASSERTION.search(clause):
            customer_premise = True
            continue
        if ((not other_person or _OWN_COSTS.search(clause))
                and _has_current_funding_assertion(clause)):
            customer_premise = True
            continue
        if _NEGATED_FUNDING.search(clause):
            continue
        if _OTHER_PERSON_COSTS.search(clause) and not _OWN_COSTS.search(clause):
            continue
        if ((source_spans or re.search(r"\b(?:money|income|support|help|allowance)\b", clause))
                and _CUSTOMER_RECEIPT.search(clause)):
            customer_premise = True
            continue
        # In the bounded personal question, an unowned 'support from ...'
        # fragment describes the customer's receipt. A person mentioned before
        # the phrase instead assigns that receipt to a third-party subject;
        # donor finances alone cannot supply the customer's funding premise.
        if any(not _THIRD_PARTY_FUNDING_CONTEXT.search(clause[:fragment.start()])
               for fragment in _CONTEXTUAL_SUPPORT_FRAGMENT.finditer(clause)):
            customer_premise = True
            continue
        for start, _ in source_spans:
            prefix = clause[:start]
            if not other_person or re.search(r"\b(?:my|our)(?: own)?\s*$", prefix):
                customer_premise = True
                break
    return other_person_context and not customer_premise


def _withdrawal_candidate_contradicted(text: str) -> bool:
    """Veto narrow ownership/use contradictions, without choosing a source.

    A donor's savings are not evidence of the customer's own withdrawals.
    A separate current own drawdown prevents an incidental donor mention from
    being a veto. Terse source labels retain the bounded question's context.
    """
    from ai.d5_text_classifier import source_mentions

    other_resource = False
    own_current_drawdown = False
    denied_resources: set[str] = set()
    own_drawdown_resources: set[str] = set()
    for clause in _clauses(_without_quoted_material(text)):
        if _NON_CURRENT.search(clause) or _HYPOTHETICAL.search(clause):
            continue
        for denial in _OWN_WITHDRAWAL_DENIAL.finditer(clause):
            denied_resources.update(match.group().rstrip("s")
                                    for match in _WITHDRAWAL_RESOURCE_WORD.finditer(denial.group()))
        if _OTHER_WITHDRAWAL_RESOURCE.search(clause) or (
            _OTHER_SUBJECT.search(clause)
            and any(mention.source == "withdrawals" for mention in source_mentions(clause))
        ):
            other_resource = True
        own_action = _OWN_WITHDRAWAL_ACTION.search(clause)
        if own_action:
            prefix = clause[max(0, own_action.start() - 35):own_action.start()]
            if not _NEGATED_FUNDING.search(prefix + own_action.group()):
                own_current_drawdown = True
                own_drawdown_resources.update(match.group().rstrip("s")
                    for match in _WITHDRAWAL_RESOURCE_WORD.finditer(own_action.group()))
    if denied_resources:
        # Savings remaining untouched does not deny current investment sales.
        # Opposing statements about the same resource remain unresolved; only
        # an explicit correction (handled earlier) supersedes that conflict.
        return bool(denied_resources & own_drawdown_resources) or not bool(
            own_drawdown_resources - denied_resources)
    return other_resource and not own_current_drawdown


def _uncertain_current_meaning(text: str) -> bool:
    for clause in _clauses(text):
        match = _UNCERTAINTY.search(clause)
        if match:
            tail = clause[match.end():].strip(" ,:-")
            # D5 identifies type, not amount, sufficiency, predictability, or a
            # later payment. Uncertainty in those separate variables is not a veto.
            if re.match(r"(?:how much|about (?:the )?(?:amount|size|adequacy|predictability)|whether (?:it|that|this) (?:is|will be) enough)\b", tail):
                continue
            if re.match(r"(?:whether|if)\b.{0,65}\b(?:enough|sufficient|adequate|covers? all|covers? everything)\b", tail):
                continue
            if _DATE_TIMING.search(tail) and not _CURRENT_SUBJECT.search(tail):
                continue
            if _NON_CURRENT.search(clause) and not _CURRENT_SUBJECT.search(tail):
                continue
            if _OTHER_OWNER.search(clause) and not _OWN_COSTS.search(tail):
                continue
            if _CURRENT_SUBJECT.search(clause) or len(tail.split()) <= 7 or re.match(r"(?:what|which|whether|if)\b", tail):
                return True
        # A short qualified answer concerns the candidate itself. A hedge in a
        # separately scoped future/inactive clause does not qualify current funding.
        if re.match(r"^(?:probably|maybe|perhaps|i guess|i think it might be)\b", clause):
            if not _NON_CURRENT.search(clause):
                return True
    return False


def _competing_exclusive_claims(text: str) -> bool:
    claims = set()
    for match in _EXCLUSIVE_CLAIM.finditer(text):
        claim = match.group("claim").strip(" ,")
        if _NON_CURRENT.search(claim) or re.search(r"\b(?:not|isn't|aren't|don't say|do not say|reject)\b", claim):
            continue
        # Different payment areas can each have a sole payer. This guard is for
        # full-source exclusivity, not 'only pays my food' vs 'only pays my travel'.
        if re.search(r"\b(?:for|of) (?:my )?(?:food|groceries|transport|heating|accommodation)\b", claim):
            continue
        claim = re.sub(r"^(?:and|also|currently|now)\s+", "", claim)
        # An anaphoric repetition adds no second exclusive account. Ordinary
        # salary/wage wording can also repeat the same account; this is narrow
        # claim equality, not a financial category or active-source mapper.
        if re.fullmatch(r"(?:it|this|that|this source|that source|the same source)", claim):
            continue
        claim = re.sub(r"\b(?:salary|wages?|pay packet|pay cheque|paycheck)\b", "employment pay", claim)
        claims.add(claim)
    return len(claims) > 1


def _without_quoted_material(text: str) -> str:
    return re.sub(r'"[^"\n]*"|(?<!\w)\'[^\'\n]*\'(?!\w)', "", text)


def _current_no_source_conflict(text: str) -> bool:
    """Veto opposing present-tense funding premises, without naming sources."""
    clauses = [part.strip() for clause in _clauses(_without_quoted_material(text)) for part in re.split(r"\band\b", clause) if part.strip()]
    no_source_claim = False
    positive_funding_claim = False
    for clause in clauses:
        if _NON_CURRENT.search(clause) or _HYPOTHETICAL.search(clause):
            continue
        if _NO_REGULAR_ASSERTION.search(clause):
            # An exception is a qualified claim, rather than explicit no source.
            if not re.search(r"\b(?:apart from|except|other than|unless|used to|was|had)\b", clause):
                no_source_claim = True
            continue
        if _ONE_OFF_RESOURCE.search(clause) or (_OTHER_OWNER.search(clause) and not _OWN_COSTS.search(clause)):
            continue
        for funding in _FUNDING_ASSERTION.finditer(clause):
            # 'regular source' alone is not an assertion of money paying costs.
            if re.fullmatch(r"(?:current|regular) (?:funding|source)", funding.group()):
                continue
            prefix = clause[max(0, funding.start() - 45):funding.start()]
            if re.search(r"\b(?:don't|do not|doesn't|does not|never|not|isn't|is not|aren't|are not|no|none|nothing)\b", prefix):
                continue
            if re.search(r"\b(?:nothing|none|no money|not|never|used to|was|were)\b", funding.group()):
                continue
            positive_funding_claim = True
    return no_source_claim and positive_funding_claim


def _dependency_flags(text: str) -> frozenset[str]:
    if not _SAME_MONEY.search(text):
        return frozenset()
    for clause in _clauses(text):
        action = _DEPENDENCY_ACTION.search(clause)
        if not action or _NON_CURRENT.search(clause):
            continue
        prefix = clause[max(0, action.start() - 35):action.start()]
        if re.search(r"\b(?:don't|do not|never|not|without|avoid|avoiding)\b", prefix):
            continue
        if re.search(r"\b(?:nothing|none|no money|not|never)\b", action.group()):
            continue
        return frozenset({"same_money_dependency"})
    return frozenset()


def _has_support_lexical_premise(text: str) -> bool:
    """A necessary support mention only; never infer its current contribution."""
    from ai.d5_text_classifier import source_mentions

    return (text == "d5_benefits_support"
            or any(mention.source == "support" for mention in source_mentions(text))
            or bool(_SUPPORT_PREMISE.search(text)))


def validate_d5_candidate(
    spec: BusinessSpec,
    text: str,
    candidate_option_id: str | None,
    previous_details: dict[str, str] | None = None,
    *, grounded_source_facts: list[dict[str, str]] | None = None,
    grounding_policy: str = "d5-llm-grounding-v3.1",
) -> BoundedAnswer:
    """Return the same ML candidate or a null veto, never a replacement option.

    A candidate is mandatory even for a terse exact source label. These checks
    do not provide a fallback when the model abstains, and deliberately do not
    attempt to certify which source category a clear sentence describes.
    """
    details = dict(previous_details or {})
    current_raw = _current_answer(text)
    current = _normalise(current_raw)
    flags = _dependency_flags(current)

    def veto(reason: str, *, bucket: str | None = "Undecided", prompt: str = _FUNDING_PROMPT) -> BoundedAnswer:
        veto_flags = flags | (frozenset({"material_conflict"}) if reason == "competing_meanings" else frozenset())
        return BoundedAnswer(bucket, reason=reason, followup=prompt, context_details=details, flags=veto_flags)

    if candidate_option_id is None:
        return veto("ml_candidate_required", bucket="Confusion")
    try:
        spec.option("D5", candidate_option_id)
    except KeyError:
        return veto("invalid_d5_candidate", bucket="Confusion")
    if not current:
        return veto("empty_reply", bucket="Confusion")
    nonmaterial_grounded_uncertainty = False
    if grounded_source_facts is not None:
        if grounding_policy == "d5-llm-grounding-v5.1":
            from ai.d5_llm_grounding_v5 import grounding_violations, only_nonmaterial_uncertainty
        elif grounding_policy == "d5-llm-grounding-v3.1":
            from ai.d5_llm_grounding import grounding_violations, only_nonmaterial_uncertainty
        else:
            return veto("unsupported_grounding_policy", bucket="Confusion")
        nonmaterial_grounded_uncertainty = (not grounding_violations(current_raw, grounded_source_facts, candidate_option_id)
                                            and only_nonmaterial_uncertainty(current_raw))
    if _uncertain_current_meaning(current) and not nonmaterial_grounded_uncertainty:
        return veto("explicit_uncertainty", prompt="Which part of the current living-cost funding is uncertain? It can remain unresolved.")
    if _competing_exclusive_claims(current):
        return veto("competing_meanings", prompt="You gave different accounts of the only current source. Which account describes your living costs now?")
    if _current_no_source_conflict(current):
        return veto("competing_meanings", prompt="You said there is no regular source and also that money currently funds your living costs. Which account applies now?")
    if candidate_option_id == "D5_WITHDRAWALS" and _withdrawal_candidate_contradicted(current):
        # This contradicts the model's candidate; it does not establish that
        # the customer's reply is undecided or select a support replacement.
        return veto("candidate_withdrawal_ownership_or_use_conflict",
                    prompt="Are these withdrawals from your own savings or investments, or regular money someone sends to you?")

    current_assertion = _has_current_funding_assertion(current)
    if (_FUTURE.search(current) or _DATE_TIMING.search(current)) and not current_assertion:
        return veto("future_income_not_current", bucket="Confusion", prompt="What funds your living costs now? A future payment is separate.")
    if _HELD_PENSION.search(current) and not current_assertion and not _PENSION_PAYMENT_ASSERTION.search(current):
        return veto("holding_not_income", bucket="Confusion", prompt="A held pension pot is different from current pension payments. What funds living costs now?")
    if _other_person_only_funding_premise(current):
        return veto("candidate_customer_funding_premise_missing", bucket=None,
                    prompt="Which source or sources currently pay your living costs? Please separate your own funding from the other person's.")
    if _POSSESSION_ONLY.search(current) and not current_assertion:
        return veto("current_source_not_established", prompt="Owning or receiving money doesn't establish which source currently pays your living costs. Which source actually does?")
    if _VAGUE_DETAIL_ONLY.fullmatch(current) or (candidate_option_id == "D5_MULTIPLE" and _UNNAMED_MIX_ONLY.fullmatch(current)):
        return veto("missing_source_detail", prompt="Please name the source or sources that currently provide money for living costs.")
    if candidate_option_id in {"D5_OTHER_REGULAR", "D5_MULTIPLE"}:
        for clause in _clauses(current):
            if _UNNAMED_CURRENT_DETAIL.search(clause) and not _NON_CURRENT.search(clause):
                return veto("missing_source_detail", prompt="Please name the other source currently paying living costs; I won't guess what category it belongs to.")
    grounded_support = (grounded_source_facts is not None
        and grounding_policy == "d5-llm-grounding-v5.1"
        and not grounding_violations(current_raw, grounded_source_facts, candidate_option_id)
        and any(f.get("source_category") == "D5_BENEFITS_SUPPORT"
            and f.get("owner") == "self" and f.get("time") == "current"
            and f.get("pays_living_costs") == "yes" for f in grounded_source_facts))
    if candidate_option_id == "D5_BENEFITS_SUPPORT" and not _has_support_lexical_premise(current) and not grounded_support:
        # The ML proposal cannot be justified with zero benefits/support
        # wording. This does not inspect any other source or select a substitute.
        # Structural model uncertainty stays outside the semantic-veto allowlist.
        return veto("candidate_source_lexical_premise_missing",
                    prompt="I couldn't support the proposed benefits or support meaning from your reply. What currently provides your living-cost money?")
    if candidate_option_id == "D5_MULTIPLE":
        # A mixed candidate needs at least two *named lexical channels*. This
        # necessary structure is not proof that either mention contributes:
        # negated, past, future and other-person mentions remain mentions. The
        # fitted model and calibrated gate must decide those meanings.
        from ai.d5_text_classifier import source_mentions

        named_channels = {mention.source for mention in source_mentions(current_raw)}
        # Workplace pay is another employment-pay *mention*. This local
        # necessary-premise alias does not change ML features, prove that the
        # pay contributes, or choose an answer after model abstention.
        if _WORKPLACE_PAY_MENTION.search(current):
            named_channels.add("salary")
        grounded_multiple = False
        if grounded_source_facts is not None:
            if grounding_violations(current_raw, grounded_source_facts, candidate_option_id):
                return veto("llm_grounded_multiple_premise_failed")
            grounded_multiple = len({f.get("source_category") for f in grounded_source_facts
                if f.get("owner") == "self" and f.get("time") == "current"
                and f.get("pays_living_costs") == "yes"
                and f.get("source_category") not in {None, "D5_MULTIPLE", "D5_NO_REGULAR"}}) >= 2
        if len(named_channels) < 2 and not grounded_multiple:
            return veto("named_sources_for_multiple_missing", prompt="Please name at least two different source categories that currently fund living costs.")

    income_detail = supported_detail(current_raw)
    if income_detail:
        details["income_detail"] = income_detail
    return BoundedAnswer(
        "Clarity", candidate_option_id, "ml_candidate_passed_d5_vetoes",
        context_details=details, flags=flags,
    )
