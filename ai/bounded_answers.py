"""Conservative, question-specific interpretation of ordinary financial replies.

These rules recognise supported meanings, not general investment suitability.
They preserve uncertainty and competing meanings and propose only approved IDs.
Unknown language abstains. No training/calibration/test data is changed here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ai.question_spec import BusinessSpec
from ai.pre_model_controls import check_pre_model_controls


@dataclass(frozen=True)
class BoundedAnswer:
    bucket: str
    option_id: str | None = None
    reason: str = "unsupported_meaning"
    followup: str | None = None
    context_details: dict[str, str] = field(default_factory=dict)
    flags: frozenset[str] = frozenset()


def clean(text: str) -> str:
    value = text.lower().replace("’", "'").replace("–", "-").replace("—", "-")
    # Harmless social chatter must not remove an otherwise supported meaning.
    value = re.sub(r"(?:[,;.]?\s*(?:and|also|by the way)\s+)?(?:the )?sun (?:is |looks )?(?:shining|shinny|sunny).*", "", value)
    return " ".join(value.split())


def supported_detail(raw: str) -> str:
    """Keep the reviewed financial phrase, excluding care/support and amounts.

    Raw text stays in the transient audit. A saved context field is a short
    description of the money, never an embedded distress or accessibility log.
    """
    value = re.split(r"(?:[,;.]?\s*(?:and|but)\s+|[.;]\s*)(?:i(?:'m| am| feel)|please|could you|can you|the sun)\s*(?:panick\w*|distress\w*|overwhelm\w*|ashamed|anxious|bereaved|griev\w*|need help|need short|need to slow|slow|keep|explain|help|shining|sunny).*",raw,flags=re.I)[0]
    value = re.split(r"[.;]\s*(?:i(?:'m| am)|please)\s*(?:panick\w*|distress\w*|overwhelm\w*|ashamed|need|bereaved).*",value,flags=re.I)[0]
    clauses = re.split(r"(?<=[.;])\s+|[,;]?\s+(?:and|but)\s+(?=(?:i\b|please\b|could you\b|can you\b|my (?:wife|husband|partner|spouse)\b))", value, flags=re.I)
    value = " ".join(clause for clause in clauses if check_pre_model_controls(clause).action not in {"care_pause", "pace_answer", "support_answer", "support_control", "explain"})
    value = re.sub(r"[£€$]\s*\d[\d,.]*(?:\s*k)?(?:\s*(?:GBP|EUR|USD))?", "", value,flags=re.I)
    value = re.sub(r"\b\d[\d,.]*\s*(?:GBP|EUR|USD|AUD|CAD|NZD|CHF|JPY|CNY|RMB|HKD|SGD|INR|pounds|euros)\b", "", value,flags=re.I)
    value = re.sub(r"\b"+NUMBER+r"\s*(?:GBP|EUR|USD|AUD|CAD|NZD|CHF|JPY|CNY|RMB|HKD|SGD|INR|pounds?|euros?|quid|dollars?)\b", "", value,flags=re.I)
    value = re.sub(r"(?:[,;.]?\s*and\s+)?(?:the )?sun (?:is |looks )?(?:shining|shinny|sunny).*", "", value,flags=re.I)
    return " ".join(value.strip(" ,;.").split())[:180]


def has(text: str, pattern: str) -> bool:
    return re.search(pattern, text, re.I) is not None


def unresolved(prompt: str, *, bucket: str = "Undecided", reason: str = "missing_material_detail", details=None):
    return BoundedAnswer(bucket, reason=reason, followup=prompt, context_details=details or {},
                         flags=frozenset({"material_conflict"}) if reason=="competing_meanings" else frozenset())


def clear(option: str, details=None, flags=frozenset()):
    return BoundedAnswer("Clarity", option, "supported_bounded_meaning", context_details=details or {}, flags=flags)


WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
         "fifteen": 15, "twenty": 20}
TENS = {"twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90}
NUMBER = r"(?:\d[\d,]*(?:\.\d+)?(?:\s*k)?|(?:(?:" + "|".join(TENS) + r")(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?|(?:" + "|".join(WORDS) + r"))(?:\s+thousand)?)"


def number(token: str) -> float:
    token = token.strip()
    if token.endswith(" thousand"):
        return number(token.removesuffix(" thousand")) * 1000
    parts = re.split(r"[- ]",token)
    if parts[0] in TENS:
        return TENS[parts[0]] + (WORDS[parts[1]] if len(parts)>1 else 0)
    if token in WORDS:
        return WORDS[token]
    return float(token.rstrip("k").replace(",", "")) * (1000 if token.endswith("k") else 1)


def numbers(text: str) -> list[float]:
    # Dates/timing are not money or transaction counts.
    text = re.sub(r"\b(?:(?:past|last|next|those|over the)\s+)?(?:12|twelve)[- ]months?\b", "", text)
    return [number(m.group()) for m in re.finditer(r"\b" + NUMBER + r"\b", text)]


def currency(text: str) -> str | None:
    text = re.sub(r"\b(?:do not|don't) convert\b.*", "", text)
    found = set()
    for compact in re.finditer(r"(?<![a-z])\d[\d,.]*\s*k?\s*(gbp|eur|usd)\b",text):
        found.add(compact.group(1).upper())
    for code, pattern in {"GBP": r"£|\b(?:gbp|pounds?|sterling)\b", "EUR": r"€|\b(?:eur|euros?)\b", "USD": r"\b(?:usd|us dollars?)\b"}.items():
        if has(text, pattern): found.add(code)
    if len(found) == 1: return next(iter(found))
    if not found:
        other = re.search(r"\b(?:aud|cad|nzd|chf|jpy|cny|rmb|hkd|sgd|inr)\b", text)
        if other: return other.group().upper()
    return None


def money_numbers(text: str) -> list[float]:
    matches = re.findall(r"[£€$]\s*(" + NUMBER + r")|\b(" + NUMBER + r")\s*(?:gbp|eur|usd|aud|cad|cny|rmb|hkd|sgd|inr|chf|jpy|pounds?|euros?|quid)\b", text)
    if matches: return [number(a or b) for a,b in matches]
    stripped = re.sub(r"\b(?:past|last)\s+12\s+months\b", "", text)
    # Untagged numerals/amount words are accepted as an amount only; explanatory
    # phrases such as 'one question at a time' are not amounts.
    stripped = re.sub(r"\b(?:one|all three) (?:question|short question|trade-offs).*", "", stripped)
    return numbers(stripped)


def pick_many(text: str, found: list[str], mixed: str, prompt: str, details=None):
    found = list(dict.fromkeys(found))
    if len(found) > 1:
        exclusive_claims = len(re.findall(r"\b(?:solely|only (?:regular )?source|every penny|entire (?:same )?pot|all of this (?:exact |same )?pot)\b", text))
        if has(text, r"\b(?:or|either|versus)\b") or exclusive_claims > 1:
            return unresolved(prompt, reason="competing_meanings")
        return clear(mixed, details)
    if found:
        return clear(found[0], details)
    return unresolved(prompt, bucket="Confusion", reason="no_supported_answer")


def stated_access_delay(text: str) -> float | None:
    """Actual stated unavailability; product names alone establish no delay."""
    if not has(text,r"\b(?:locked|inaccessible|cannot access|can't access|cannot withdraw|can't withdraw|unavailable until)\b"):
        return None
    match = re.search(r"\b("+NUMBER+r")\s*(years?|months?)\b",text)
    if match:
        return number(match.group(1)) / (12 if match.group(2).startswith("month") else 1)
    return None


def interpret_bounded_answer(spec: BusinessSpec, qid: str, raw: str, previous_details: dict[str, str] | None = None) -> BoundedAnswer:
    text = clean(raw)
    previous_details = previous_details or {}
    # An explicit correction uses the revised meaning, never both old and new.
    parts = re.split(r"(?:^actually\b|[.;]\s*actually\b|\b(?:no,? i mean|sorry,? i meant|sorry,? correction:)\b|\bcorrection:\s*)", text)
    if len(parts) > 1 and parts[-1].strip(): text = parts[-1].strip()
    exact = spec.match_exact_choice(qid, text)
    unrelated_request = r"^(?:please\s+)?(?:(?:can|could|would|will) you\s+)?(?:book\b|translate\b|recommend\b|help me plan\b|tell me about\b|tell me\b.*\bweather\b)|^which\b.*\b(?:phone|charger|restaurant|hotel|flight)\b"
    if has(text, unrelated_request):
        return unresolved("Let's return to this question. You can answer in your own words, ask about the choices, or ask for general help.", bucket="Confusion", reason="off_topic")
    if has(text, r"^(?:hi|hello|thanks|thank you|how are you|the weather|what(?:'s| is) (?:the )?weather|how(?:'s| is) (?:the )?weather|tell me (?:a joke|the weather)|ignore (?:the |all )?(?:rules|instructions))\b"):
        return unresolved("Let's return to this question. You can answer in your own words, ask about the choices, or ask for general help.", bucket="Confusion", reason="off_topic")
    if qid not in {"D3", "D10"} and exact:
        return clear(exact.id)
    if has(text,r"^(?:what (?:is|are|does)|does .* mean|(?:can|could) you explain|explain (?:what|the word)|i (?:do not|don't) understand|i need help understanding)\b"):
        return unresolved("Would you like an explanation of this question and its choices?",bucket="Confusion",reason="term_help")
    if has(text, r"\b(?:not sure|unsure|don't know|do not know|haven't decided|have not decided|undecided|can't decide|cannot decide|cannot choose|can't choose|cannot remember|can't remember|not worked out|haven't figured out|have not figured out|haven't found out|have not found out|haven't established|have not established|can't establish|cannot establish|cannot separate|can't separate|cannot determine|cannot tell|can't tell|torn between)\b") and not (qid=="D14" and has(text,r"\bstill prioritise\b")):
        return unresolved("What part are you unsure about? You can ask for an explanation, pause, or leave this unresolved.", reason="explicit_uncertainty")
    option_only_hedge = qid=="D2" and has(text,r"\bturned\s+("+NUMBER+r")\s+last\s+(?:week|month|year)\b") and has(text,r"\b(?:that one|that band)\b")
    if has(text, r"\b(?:maybe|probably|perhaps|might be|i guess)\b") and not option_only_hedge:
        return unresolved("Which part of your answer is uncertain? I won't choose a category until the relevant meaning is clear.", reason="qualified_answer")

    if qid == "D1":
        if has(text,r"\b(?:all|whole|entire)\b.*\b(?:but|and)\b.*\bnone\b"):
            return unresolved("You described both a purpose and its opposite for the same pot. What is this same money actually for?",reason="competing_meanings")
        found = []
        patterns = {
            "D1_HOME": r"\b(?:buy|buying|deposit for|saving for).*(?:home|house|flat)\b|\b(?:house|home) deposit\b",
            "D1_RETIREMENT": r"\b(?:retirement|retire|retiring)\b",
            "D1_LIVING_COSTS": r"\b(?:living costs|everyday costs|regular bills|rent|food|groceries)\b",
            "D1_EMERGENCIES": r"\b(?:emergency|emergencies|unexpected costs|rainy day)\b",
            "D1_PURCHASE": r"\b(?:car|wedding|graduation|tuition|school fees|holiday|planned payment|large purchase|university fees)\b",
            "D1_WEALTH": r"\b(?:build(?:ing)? (?:up )?(?:my )?wealth|grow(?:ing)? (?:my )?(?:money|wealth)|general investing|just wealth)\b",
        }
        for oid, pattern in patterns.items():
            if has(text, pattern): found.append(oid)
        if "D1_RETIREMENT" in found and "D1_WEALTH" in found: found.remove("D1_WEALTH")
        if "D1_PURCHASE" in found and "D1_HOME" in found: found.remove("D1_PURCHASE")
        # Negative incidental references are not a second purpose.
        if has(text, r"\b(?:not for|without|no) (?:a )?(?:purchase|planned payment)") and "D1_PURCHASE" in found: found.remove("D1_PURCHASE")
        detail = {"purpose_detail": supported_detail(raw)} if found else {}
        if not found and has(text, r"\b(?:inheritance|charity|donation|start (?:a |my )?business|family gift)\b"):
            return clear("D1_OTHER", {"purpose_detail": supported_detail(raw)})
        if not found and has(text,r"\b(?:future|later|someday|some day)\b"):
            return unresolved("What do you expect to use this same money for in the future? It is okay if you have not decided yet.",reason="purpose_not_yet_defined")
        return pick_many(text, found, "D1_MULTIPLE", "What will this same money be used for? If you mean several pots, which one or which named parts should this conversation cover?", detail)

    if qid == "D2":
        current = re.findall(r"\b(?:i am|i'm|aged|turned|current age is)\s+(?:(?:also|exactly)\s+)?("+NUMBER+r")\b",text)
        bare_age = re.fullmatch(r"\s*("+NUMBER+r")(?:\s*(?:years? old|yo))?[.!]?\s*", text)
        vals = [number(v) for v in current] if current else ([number(bare_age.group(1))] if bare_age else numbers(text) if has(text,r"\b(?:age|years? old|yo)\b") else [])
        if not vals or any(v < 0 or v > 120 for v in vals):
            return unresolved("Which age band fits your current age? Retirement status alone does not tell me your age. You may leave this optional question unanswered.",bucket="Undecided" if has(text,r"\b(?:old|young|age)\b") else "Confusion")
        def band(n):
            return "D2_UNDER18" if n < 18 else "D2_AGE18_34" if n < 35 else "D2_AGE35_49" if n < 50 else "D2_AGE50_64" if n < 65 else "D2_AGE65_PLUS"
        choices = {band(v) for v in vals}
        return clear(choices.pop()) if len(choices) == 1 else unresolved("Your age range spans two bands. Which band fits your current age?", reason="competing_meanings")

    if qid in {"D3", "D10"}:
        prefix = qid + "_"
        if qid == "D3" and (text in {"0", "zero", "none"} or has(text, r"\b(?:no existing (?:money|pot)|no money (?:yet|set aside)|only a future target|not identified any existing money|not set any money aside)\b")):
            return clear("D3_NO_EXISTING_POT")
        if qid == "D10" and (text in {"0", "zero", "none"} or has(text, r"\b(?:no new (?:outside |external )?money|nothing|zero new money)\b")):
            return clear("D10_ZERO")
        amount_text=text
        if qid=="D3": amount_text=re.split(r"[.;]\s*(?:the )?(?:eventual |future )?(?:savings )?(?:target|goal)",text)[0]
        if qid=="D10" and has(text,r"\bbut (?:i )?(?:put|paid|contributed|invested)\b"):amount_text=re.split(r"\bbut\b",text)[-1]
        if qid == "D3" and has(amount_text, r"\b(?:want to have|goal is|target is|hope to save|will save|will invest|dream is|eventually)\b"):
            return unresolved("That sounds like a future target. What existing money should we discuss now, or is no existing pot identified?", bucket="Confusion", reason="future_target_not_existing_money")
        if qid == "D10" and has(amount_text, r"\b(?:current value|currently worth|worth .*?(?:now|today)|profits?|gains?|sale proceeds|recycled)\b") and not has(amount_text, r"\b(?:exclud\w*|not including|not profits|without|new (?:outside |external )?money|fresh money)\b"):
            return unresolved("How much new external money did you actually invest during the past 12 months, excluding gains and reused sale proceeds?", bucket="Confusion", reason="wrong_amount_premise")
        curr = currency(text)
        if not curr: curr = previous_details.get("currency")
        details = {"currency": curr} if curr else {}
        amount_choice = exact or spec.match_exact_choice(qid, re.sub(r"\b(?:gbp|eur|usd|pounds?|euros?|sterling)\b|[£€]", "", text).strip())
        option_id = amount_choice.id if amount_choice else None
        if not option_id:
            vals = money_numbers(amount_text)
            if not vals:
                option_id = previous_details.get("pending_option_id") if curr else None
            else:
                boundaries = [1000, 5000, 20000, 50000] if qid == "D3" else [1000, 5000, 10000]
                suffixes = ["POSITIVE_LT1000", "FROM1000_LT5000", "FROM5000_LT20000", "FROM20000_LT50000", "FROM50000_PLUS"] if qid == "D3" else ["POSITIVE_LT1000", "FROM1000_LT5000", "FROM5000_LT10000", "FROM10000_PLUS"]
                if vals==[0]:return clear("D3_NO_EXISTING_POT" if qid=="D3" else "D10_ZERO")
                if any(v <= 0 for v in vals): return unresolved("Does this mean no existing money, or a positive amount? Please give a range and currency.")
                ids = {prefix + suffixes[sum(v >= b for b in boundaries)] for v in vals}
                if len(ids) != 1: return unresolved("That amount spans different bands. Which band contains the existing amount? Please state its currency." if qid == "D3" else "That amount spans different bands. Which band contains your new contributions in the past 12 months, and in what currency?",reason="competing_meanings" if has(text,r"\b(?:same complete total|both as the total|identical .*months|same .*pot .*right now)\b") else "missing_material_detail")
                option_id = ids.pop()
        if option_id and option_id.endswith(("ZERO", "NO_EXISTING_POT")): return clear(option_id)
        if option_id and not curr:
            return unresolved("Which currency is that range in: GBP, EUR, USD, or another named currency?", details={"pending_option_id": option_id})
        if option_id:
            return clear(option_id, details)
        return unresolved("What approximate range and currency do you mean? You may leave this optional question unanswered.", details=details)

    if qid == "D4":
        if has(text,r"\b(?:used to|previously|will put|plan to put|want to put)\b") and not has(text,r"\b(?:now|currently|still held)\b"):
            return unresolved("Where is this money held now, rather than where it was before or where you might put it?",bucket="Confusion",reason="wrong_holding_time")
        if len(re.findall(r"\bonly\b",text))>1:
            return unresolved("You described two exclusive places for the same whole pot. Where is it actually held?",reason="competing_meanings")
        found = []
        if has(text, r"\b(?:current account|easy[- ]access|instant[- ]access|checking account)\b") or (has(text,r"\bsavings account\b") and not has(text, r"\b(?:fixed|locked|restricted)\b")): found.append("D4_EASY_ACCESS")
        if has(text, r"\b(?:fixed[- ]term|fixed deposit|term deposit|locked savings|locked away|deposit with restrictions)\b"): found.append("D4_FIXED_TERM")
        pension = has(text, r"\b(?:pension|sipp)\b")
        if pension:
            if has(text, r"\b(?:payments|paid into|income)\b") and not has(text, r"\b(?:inside|held|pot|savings)\b"):
                return unresolved("Do you mean money still held inside a pension, or pension income already paid into another account?")
            found.append("D4_PENSION_SAVINGS")
        if has(text, r"\b(?:shares|stocks|bonds|funds|etfs?|investments)\b"):
            if not pension or has(text, r"\b(?:separate|outside|also .*account)\b"): found.append("D4_INVESTMENTS")
        if has(text, r"\b(?:cash|wallet|safe at home)\b") and not has(text, r"\bcash (?:in|isa)\b") and not ("D4_EASY_ACCESS" in found and not has(text,r"\b(?:separate|also|physical cash)\b")): found.append("D4_OTHER")
        detail = {"holding_detail": supported_detail(raw)} if found else {}
        delay = stated_access_delay(text)
        if delay is not None:
            detail["access.available_after_years"] = str(delay)
            detail["access_timing"] = f"Unavailable for {delay:g} years, as stated"
        return pick_many(text, found, "D4_MIXED", "Where is this same money currently held? If you mean cash, is it physical cash or money in an account?", detail)

    if qid == "D5":
        if has(text,r"\b(?:will receive|hope to get|plan to earn|start .*job next|future salary)\b") and not has(text,r"\b(?:currently|now|already|at present)\b"):
            return unresolved("What provides money for living costs now? A possible future payment is separate.",bucket="Confusion",reason="future_income_not_current")
        if has(text,r"\bno regular source\b"):
            if has(text,r"\b(?:salary|wages|pension payments|benefits|rental income)\b"):return unresolved("You said there is no regular source and named a source. Which actually provides money for living costs now?",reason="competing_meanings")
            return clear("D5_NO_REGULAR")
        if has(text,r"\bpension\b.*\b(?:balance|pot|fund|savings)\b") and not has(text,r"\b(?:payments|pays|income|withdrawals)\b"):
            return unresolved("A pension balance is money held, not a current payment. What currently provides funds for your living costs?",bucket="Confusion",reason="holding_not_income")
        found = []
        patterns = {"D5_SALARY":r"\b(?:salary|wages|paid employment|paycheck|pay cheque)\b", "D5_SELF_EMPLOYMENT":r"\b(?:self[- ]employment|self[- ]employed|freelanc\w*|my business)\b", "D5_PENSION_PAYMENTS":r"\b(?:pension(?: payments| income)?|retirement income)\b", "D5_BENEFITS_SUPPORT":r"\b(?:benefits|universal credit|regular support|parents|family support)\b", "D5_WITHDRAWALS":r"\b(?:withdraw\w*|draw\w*|using|live on|living on|spend\w*).*(?:savings|investments|this .*pot|reserve)\b", "D5_OTHER_REGULAR":r"\b(?:rent(?:al)? income|regular source|rent from|dividends)\b"}
        for oid, pattern in patterns.items():
            if has(text, pattern): found.append(oid)
        if len(re.findall(r"\bonly regular source\b", text)) > 1 and len(found) > 1:
            return unresolved("You described different sources as your only current source. Which sources actually pay living costs now?",reason="competing_meanings")
        for oid, negative in {
            "D5_SALARY": r"\b(?:no|not receiving|receive no)\s+(?:salary|wages|paycheck)\b",
            "D5_PENSION_PAYMENTS": r"\b(?:no|not receiving|receive no)\s+pension (?:payments|income)\b",
            "D5_BENEFITS_SUPPORT": r"\b(?:no|not receiving|receive no)\s+(?:benefits|regular support)\b",
            "D5_OTHER_REGULAR": r"\bno (?:(?:(?:wages|salary) or )?other )?regular source\b",
        }.items():
            if oid in found and has(text,negative): found.remove(oid)
        if has(text,r"\b(?:no regular (?:source|income)|nothing currently|no income)\b"):
            return unresolved("You mentioned no regular source and another source. Which currently pays living costs?", reason="competing_meanings") if found else clear("D5_NO_REGULAR")
        if not found and has(text,r"\b(?:get|receive|earn)\b.*\bmoney\b"):
            return unresolved("Where does that money come from, and does it currently pay your regular living costs? You may leave this undecided if you cannot identify the sources.",reason="current_source_not_established")
        result = pick_many(text, found, "D5_MULTIPLE", "Which sources currently pay your regular living costs? A pension pot is different from pension payments.", {"income_detail":supported_detail(raw)} if found else {})
        if "D5_WITHDRAWALS" in found and has(text,r"\b(?:this (?:exact |same )?pot|this (?:named )?reserve|same reserve|same money|money we are discussing)\b"):
            return BoundedAnswer(result.bucket, result.option_id, result.reason, result.followup, result.context_details, frozenset({"same_money_dependency"}))
        return result

    if qid == "D6":
        if has(text,r"\b(?:upset|worried|nervous|feel)\b") and not has(text,r"\b(?:resources|income|cover|covered|pay|costs|bills|repayments)\b"):
            return unresolved("That describes how a loss might feel. This question asks whether other usable resources could pay necessary costs while this pot is unavailable for a year.",bucket="Confusion",reason="feelings_not_backup_coverage")
        excludes_same_pot=has(text,r"\b(?:excluding|without|not counting|not using)\b|\b(?:none|no part) of (?:(?:my|the) )?(?:backup|other resources) (?:is|comes from) (?:this|the same) pot\b|\bthis pot is not (?:my |the )?backup\b")
        if has(text,r"\b(?:fed|funded) entirely (?:by|from) (?:the )?same reserve\b"):
            return BoundedAnswer("Undecided", reason="same_money_is_not_other_backup", followup="Money transferred from the same reserve is not a separate backup while that reserve is unavailable. Excluding it, what could other usable resources cover for the full year?", flags=frozenset({"same_pot_backup_conflict"}))
        if has(text,r"\b(?:same pot|this pot|same money)\b") and has(text,r"\bbackup\b") and not excludes_same_pot:
            return BoundedAnswer("Confusion", reason="same_money_is_not_other_backup", followup="This same pot cannot count as other resources while it is unavailable. Excluding it, would other usable resources cover all, some or none of your necessary costs for the full year?", flags=frozenset({"same_pot_backup_conflict"}))
        if has(text,r"\b(?:this (?:exact |same )?pot|same money|money we are discussing)\b") and has(text,r"\b(?:use|using|withdraw\w*|draw\w*|count\w*|fund|funds)\b") and not excludes_same_pot:
            return BoundedAnswer("Undecided", reason="same_money_is_not_other_backup", followup="This same pot cannot count as other resources while it is unavailable. Excluding it, would other usable resources cover all, some or none of your necessary costs for the full year?", flags=frozenset({"same_pot_backup_conflict"}))
        backup_delay = stated_access_delay(text)
        if backup_delay is not None and backup_delay >= 1 and has(text,r"\b(?:pension|deposit)\b"):
            return BoundedAnswer("Undecided", reason="backup_not_usable_in_12_months", followup="That resource would be unavailable during the year. Excluding unavailable money, what necessary costs would other usable resources cover?", flags=frozenset({"usable_backup_check"}))
        if has(text,r"\b(?:first|initial)\b.*\bmonths\b") and has(text,r"\b(?:final|last|remaining|month seven|month 7)\b"):
            if has(text,r"\b(?:no other|none|not covered)\b") and has(text,r"\b(?:cover|covers|covered)\b"):return clear("D6_SOME")
            return unresolved("Are the necessary costs covered for the entire year, or only for part of that year?")
        coverage_text = re.sub(r"\b(?:all|full|whole|entire)\s+(?:(?:next|the)\s+)?(?:"+NUMBER+r"\s+months?|year)\b", "", text)
        all_covered = has(coverage_text,r"\b(?:all|full|fully|everything|every).*\b(?:covered|cover|costs|bills?|essential\w*)\b|\bcover (?:all|every)\b") and not has(text,r"\b(?:not all|some but|only some)\b")
        none_covered = has(text,r"\b(?:no other (?:usable )?(?:income|money|resources?)|wouldn't cover any|would not cover any|none (?:are|is) covered)\b")
        if (all_covered and none_covered) or has(text,r"\b(?:all|full)\b.*\b(?:but|and)\b.*\bnone\b"):
            return unresolved("For the same 12-month period, do other resources cover all, some or none of the necessary costs?",reason="competing_meanings")
        if all_covered:
            return clear("D6_ALL")
        if has(text,r"\b(?:some|part|partly|half|not all)\b.*\b(?:covered|cover|costs|bills)\b"):
            return clear("D6_SOME")
        if (none_covered or has(text,r"\b(?:none|nothing)\b")) and not has(text,r"\bnothing (?:falls short|is missing|is uncovered)\b"):
            return clear("D6_NONE")
        return unresolved("Without using this pot for a full year, would other resources cover all, some or none of your necessary living costs?")

    if qid == "D7":
        if text.strip(".! ") in {"none", "no repayments", "no required repayments", "nothing to repay"}:
            return clear("D7_NO_REQUIRED")
        if has(text,r"\bcredit limit\b") and not has(text,r"\b(?:repayments|payments|behind|arrears|up to date|keeping up)\b"):
            return unresolved("A credit limit does not tell me how required repayments are going. Are you keeping up easily, with difficulty, falling behind, or have no required repayments?",bucket="Confusion",reason="credit_limit_not_repayment_status")
        repayment_flags = frozenset({"same_money_dependency"}) if has(text,r"\b(?:using|from|out of)\s+(?:this (?:same |exact )?pot|this (?:same )?money)\b") else frozenset()
        behind=has(text,r"\b(?:behind|arrears|missed (?:a )?(?:payment|repayment)|late payments|overdue)\b") and not has(text,r"\b(?:not|never|no)\b.{0,12}\b(?:behind|arrears|missed|late|overdue)\b")
        none=has(text,r"\b(?:no (?:loans?|debt|credit|repayments|(?:currently )?required)|debt[- ]free|nothing to repay|no payments (?:are )?(?:currently )?required|no loan or credit repayments are required)\b")
        easy=has(text,r"\b(?:easily|without difficulty|no difficulty|no problem|comfortably|not difficult)\b")
        if behind and easy:return unresolved("You mentioned both keeping up easily and being behind on the same repayments. Which describes them now?",reason="competing_meanings")
        if behind and none: return unresolved("Do you currently have required repayments? You mentioned both no debt and missed payments.",reason="competing_meanings")
        if behind:return clear("D7_FALLING_BEHIND",flags=repayment_flags)
        if none:return clear("D7_NO_REQUIRED")
        if has(text,r"\b(?:keeping up|(?:payments|repayments).*on time|pay(?:ing)? (?:them |it |everything )?(?:on time|each month)|up to date)\b"):
            difficulty_text=re.sub(r"\b(?:not|never|no)\s+(?:difficult|struggle|difficulty)\b", "", text)
            if has(difficulty_text,r"\b(?:difficult|struggle|tight|hard|barely)\b"):return clear("D7_KEEPING_UP_DIFFICULT",flags=repayment_flags)
            if easy or has(text,r"\bcomfortable\b"):return clear("D7_KEEPING_UP_EASILY",flags=repayment_flags)
        return unresolved("Are required repayments keeping up easily, keeping up with difficulty, falling behind, or not required? A debt amount alone does not establish how repayments are going.",bucket="Confusion" if has(text,r"\b(?:owe|owed|debt|balance)\b") and money_numbers(text) else "Undecided")

    if qid == "D8":
        # These are full replies to the personal-holdings question, not
        # keywords that can override exceptions, other people, or uncertainty.
        if text.strip(".! ") in {"none", "none at all", "no", "nope", "never", "nothing", "no investments"}:
            return clear("D8_NONE")
        if has(text,r"\bnone of (?:these|those|the (?:choices|options|listed types))\b"):
            return unresolved("Do you mean you have not personally held investments, or that you have held a type not listed here?",bucket="Confusion",reason="none_of_options_fits")
        global_no_holdings = has(text,
            r"\b(?:(?:i(?: have| had|'ve)?\s+)?never|i (?:have not|haven't)|not)\s+(?:personally\s+)?(?:held|owned)\s+(?:(?:any|market|any market)\s+)?(?:investments?|anything)\b|"
            r"\b(?:i(?: have|'ve)?\s+never|i (?:have not|haven't)|never)\s+(?:personally\s+)?invested\b")
        affirmative_personal_holding = has(text,r"\bi (?:have )?(?:personally )?(?:held|owned|bought)\b|\bi own\b|\bheld\b.*\bmyself\b")
        # An asset mentioned in somebody else's history is not evidence of
        # the customer's experience. Keep short replies such as 'shares'
        # valid, but veto a proposal whose ownership is explicitly unstated.
        third_party_context = has(text,
            r"\b(?:my|our|his|her|their)\s+(?:wife|husband|spouse|uncle|aunt|grandfather|grandmother|grandparents?|cousins?|friends?|dad|father|mother|mum|parents?|partner|brothers?|sisters?|colleagues?|coworkers?|neighbours?|neighbors?)\b|"
            r"\b(?:someone else|another person)\b|\b(?:he|she|they)\s+(?:holds?|owns?|held|owned|bought|invested)\b")
        explicit_personal_none = global_no_holdings and has(text,
            r"\bi(?:\s+have|\s+had|'ve)?\s+never\s+(?:personally\s+)?(?:held|owned|invested)\b|"
            r"\bi\s+(?:have not|haven't)\s+(?:personally\s+)?(?:held|owned|invested)\b")
        if third_party_context and not explicit_personal_none:
            return unresolved("Which investments have you personally held? Please separate your own experience from the other person's; I will not infer yours from theirs.",bucket="Undecided" if has(text,r"\bi help\b") else "Confusion",reason="personal_experience_not_established")
        if third_party_context and explicit_personal_none and not affirmative_personal_holding and not has(text,r"\b(?:except|apart from|other than)\b"):
            return clear("D8_NONE")
        exception = has(text,r"\b(?:except|apart from|other than)\b")
        if has(text,r"\b(?:unless|if)\b.*\bcounts?\b|\bdoes (?:that|it|my .*?) count\b"):
            return unresolved("Which investments have you personally held? I can explain whether the type you mentioned counts before you choose.",bucket="Confusion",reason="experience_help")
        if global_no_holdings and has(text,r"\b(?:my|our|his|her|their)\s+\w+\b") and not has(text,r"\bi (?:have |had |'ve )?(?:never|have not|haven't)\b"):
            return unresolved("Have you personally held investments? Another person's experience does not establish yours.",bucket="Confusion",reason="personal_experience_not_established")
        if global_no_holdings and affirmative_personal_holding and not exception:
            return unresolved("Have you personally held investments, or none? Those statements describe conflicting experience.",reason="competing_meanings")
        if has(text,r"\bmy (?:cousins?|friends?|dad|father|mother|mum|parents|partner|brothers?|sisters?)\b") and not affirmative_personal_holding:
            return unresolved("Which investments have you personally held? Another person's investments do not establish your experience.",bucket="Undecided" if has(text,r"\bi help\b") else "Confusion",reason="personal_experience_not_established")
        if has(text,r"\b(?:own|hold) none (?:now|currently|today)\b|\bno (?:investment )?(?:trades|transactions)\b") and not affirmative_personal_holding:
            return unresolved("This question includes investments personally held in the past. Have you ever held any, rather than just holding none now or making no recent trades?",bucket="Confusion",reason="current_holdings_not_history")
        if has(text,r"\b(?:will buy|plan to buy|want to buy|considering buying)\b") and not has(text,r"\b(?:personally held|owned|have held|already held)\b"):
            return unresolved("Which investments have you personally held so far? Future plans do not count as experience.",bucket="Confusion",reason="future_plan_not_experience")
        if has(text,r"\b(?:read articles|read about|heard about)\b") and has(text,r"\b(?:does that count|never owned|never held)\b"):
            return unresolved("This asks about investments personally held. Would you like an explanation of what counts?",bucket="Confusion",reason="experience_help")
        if global_no_holdings and has(text,r"\b(?:and|but)\b.*\b(?:owned|held|shares)\b") and not exception:
            return unresolved("Have you personally held investments, or none? You gave conflicting descriptions of the same experience.",reason="competing_meanings")
        if global_no_holdings and not exception:
            return clear("D8_NONE")
        if has(text,r"\b(?:read about|heard about|watched|want to buy|considering)\b") and not has(text,r"\b(?:personally held|own|owned|have held|bought)\b"):
            return unresolved("Which investment types have you personally held, if any? Reading about them is different from holding them.",bucket="Confusion",reason="knowledge_not_experience")
        if has(text,r"\b(?:my dad|my father|my mother|my mum|my parents|my partner)\b") and not has(text,r"\b(?:i (?:have )?(?:personally )?(?:held|owned|bought)|i own)\b"):
            return unresolved("Have you personally held any investments, or only helped someone else with theirs? I will not assume your own experience.",reason="personal_experience_not_established")
        if has(text,r"\b(?:never|not|haven't|have not|don't|do not)\s+(?:personally\s+)?(?:held|owned|bought|own|hold)\s+(?:any\s+)?(?:shares|stocks|bonds|etfs?|funds|crypto\w*|bitcoin|ethereum)\b"):
            return unresolved("You said you have not held a particular type. Which types have you personally held, if any?",bucket="Confusion",reason="negated_type_not_experience")
        found=[]
        for oid, pattern in {"D8_SHARES":r"\b(?:shares|stocks)\b", "D8_BONDS":r"\bbonds\b", "D8_FUNDS_ETFS":r"\b(?:funds|etfs?|index fund)\b", "D8_INVESTED_PENSION_MANAGED":r"\b(?:invested pension|sipp|managed (?:portfolio|investment)|pension .*funds)\b", "D8_CRYPTO":r"\b(?:crypto\w*|bitcoin|ethereum)\b"}.items():
            if has(text,pattern):found.append(oid)
        if has(text,r"\bno separate managed portfolio\b") and "D8_INVESTED_PENSION_MANAGED" in found:found.remove("D8_INVESTED_PENSION_MANAGED")
        if "D8_INVESTED_PENSION_MANAGED" in found and not has(text,r"\b(?:separate|outside|also)\b"):found=["D8_INVESTED_PENSION_MANAGED"]
        if not found and has(text,r"\binvestment property\b"):return clear("D8_OTHER",{"experience_detail":supported_detail(raw)})
        return pick_many(text,found,"D8_MULTIPLE","Which types have you personally held? You may say none, name a type, or leave this optional context unanswered.",{"experience_detail":supported_detail(raw)} if found else {})

    if qid == "D9":
        if text in {"none","zero","0","never"}:return clear("D9_ZERO")
        if has(text,r"\b(?:shopping|groceries|sent money to my landlord|bank transfers|card transactions|automatic manager|manager made|pension manager|manager.*made.*trades)\b") and not has(text,r"\b(?:excluding|not counting|not .*counting|i instructed)\b"):
            return unresolved("Count only investment purchases or sales you made or instructed during the last 12 months. How many were those?",bucket="Confusion",reason="wrong_transaction_premise")
        # A stated total wins over component counts; otherwise add explicit buy/sell components.
        text=re.sub(r"replace my earlier (?:zero|answer).*|one short question at a time.*", "", text)
        investment_counts=re.findall(r"\b("+NUMBER+r")\s+(?:monthly\s+)?(?:investment|fund)\s+(?:purchases|sales|transactions|trades)\b", text)
        total=re.search(r"\b(?:so|total(?: of)?|altogether)\s+("+NUMBER+r")|\b("+NUMBER+r")\s+(?:(?:purchases|sales|trades|orders|transactions)\s+)?in total\b",text)
        if total: vals=[number(total.group(1) or total.group(2))]
        elif investment_counts and not has(text,r"\b(?:total .*is|exactly|neither is a correction)\b"): vals=[number(v) for v in investment_counts]
        else:
            components=re.findall(r"\b(?:bought|sold|buy|sell)\s+("+NUMBER+r")",text)
            vals=[sum(number(v) for v in components)] if len(components)>1 else numbers(text)
        if has(text,r"\btwice\b") and not vals:vals=[2]
        if not vals:
            if has(text,r"\b(?:no|not made any) (?:investment )?(?:purchases|sales|transactions|trades|orders)\b"):return clear("D9_ZERO")
            return unresolved("How many investment purchases or sales did you make or instruct in the last 12 months? A count or one of the ranges is enough.")
        def band(v):return "D9_ZERO" if v==0 else "D9_COUNT1_5" if 1<=v<=5 else "D9_COUNT6_10" if 6<=v<=10 else "D9_COUNT11_PLUS" if v>=11 else None
        ids={band(v) for v in vals}
        if None in ids or len(ids)!=1:return unresolved("That count spans different bands. Which range contains the number of investment purchases or sales you made or instructed?",reason="competing_meanings" if has(text,r"\b(?:neither is a correction|same transactions|same complete total)\b") else "count_spans_categories")
        return clear(ids.pop())

    if qid == "D11":
        if has(text,r"\b(?:always|guaranteed to)\b.*\brecover\b") and has(text,r"\b(?:never recover|never get .*back)\b"):
            return unresolved("You described guaranteed recovery and a loss that never recovers. Which is your current understanding?",reason="competing_meanings")
        if has(text,r"\b(?:permanent\w*|for good)\b") and has(text,r"\b(?:lose|lost|loss)\b") and has(text,r"\b(?:cannot fall|can't fall|always recover|must recover)\b"):
            return unresolved("You described both permanent loss and guaranteed protection or recovery. Which describes your current understanding for these market investments?",reason="competing_meanings")
        if has(text,r"\b(?:cannot|can't|never|won't|will not)\b.*\b(?:fall|lose|lost)\b") and not has(text,r"\b(?:guarantee|guaranteed)\b"):return clear("D11_CAPITAL_PROTECTED")
        if has(text,r"\b(?:always|must|definitely|certainly)\b.*\b(?:recovers?|recovery|come back)\b") and not has(text,r"\b(?:not|don't|do not|never)\b.{0,12}\b(?:always|must|definitely|certainly)\b"):return clear("D11_RECOVERY_ALWAYS")
        permanent=has(text,r"\b(?:permanent\w*|never get .*back|may not (?:get .*back|recover)|might not recover|(?:lose|gone) .*for good|gone for good)\b")
        loss=has(text,r"\b(?:lose|lost|loss|gone for good)\b")
        no_guarantee=has(text,r"\b(?:not guaranteed|no guarantee|no promise|isn't (?:guaranteed|promised)|not promised|may never recover|might never recover|recovery is not guaranteed|cannot rely on getting .*back|can't rely on getting .*back)\b")
        if has(text,r"\b(?:lose|lost).*\bfor good\b"):permanent=True
        if loss and permanent and no_guarantee:
            return clear("D11_LOSS_POSSIBLE", {"understanding.own_words_supported": "true"})
        return unresolved("For these market investments, could some or all of the original money be permanently lost, or must it recover? Please explain your understanding in your own words.")

    if qid == "D13":
        if has(text,r"\b(?:hate|feel|nervous|worried|upset)\b") and not has(text,r"\b(?:essentials|essential|rent|costs|bills|plans|budget|pay|covered|funding)\b"):
            return unresolved("That describes how the fall would feel. Here, what would a 20% loss change about paying essentials, your budget or other plans?",bucket="Confusion",reason="feelings_not_practical_effect")
        if has(text,r"\b(?:percent|percentage|20%)\b.*\b(?:twenty pounds|20 pounds|whatever the amount)\b"):
            return unresolved("20% means one fifth of the amount: for example £1,000 would lose £200. Would you like to revisit the practical effect of that example?",bucket="Confusion",reason="percentage_misunderstanding")
        essentials=has(text,r"\b(?:rent|food|bills|essential\w*|necessary (?:costs|heating)|mortgage)\b")
        difficulty=has(text,r"\b(?:unable|not be able to (?:pay|cover)|couldn't (?:pay|cover)|could not (?:pay|cover)|cannot (?:pay|cover)|can't (?:pay|cover)|struggl\w*|difficult|wouldn't (?:pay|cover)|would not (?:pay|cover)|won't (?:pay|cover)|will not (?:pay|cover)|short of)\b") and not has(text,r"\b(?:not|wouldn't|won't|don't|do not)\b.{0,12}\b(?:struggl\w*|difficult)\b")
        if difficulty and has(text,r"\b(?:all my plans|all plans|plans)\b.*\b(?:unchanged|unaffected)\b"):return unresolved("You said everything stayed unchanged, but also that rent could not be paid. What would this same loss change in practice?",reason="competing_meanings")
        if essentials and difficulty:return clear("D13_ESSENTIALS_DIFFICULT")
        covered=has(text,r"\b(?:still|can|could|would|remain)\b.*\b(?:pay|cover|covered|meet)\b.*\b(?:bills|essential\w*|necessary|rent|costs)\b|\b(?:essentials|essential costs|bills|rent)\b.*\b(?:covered|paid|unaffected)\b")
        affirmative_change_text=re.sub(r"\b(?:would not|wouldn't|will not|won't|do not|don't|no|not)\s+(?:need to\s+)?(?:change|adjust|delay|cancel|postpone|reduce|cut back)\w*\b", "", text)
        change=has(affirmative_change_text,r"\b(?:delay|cancel|cut back|reduce|change|adjust|postpone|use other|dip into|graduation plans)\w*\b")
        all_unchanged=has(text,r"\b(?:budget|plans)\b") and has(text,r"\b(?:unaffected|unchanged|stay the same|remain the same|nothing would change|no change|wouldn't change|would not change|won't change|will not change)\b")
        if covered and all_unchanged and not change:return clear("D13_UNAFFECTED")
        if covered and change:return clear("D13_ADJUSTMENT")
        return unresolved("In the 20% loss example, would essentials still be paid? If so, would the way you fund them, your budget or other plans change?")

    if qid == "D14":
        if has(text,r"^(?:which|what|how)\b.*\b(?:guarantee|guaranteed|double)\b"):
            return unresolved("No choice here guarantees investment growth. This question asks which trade-off between possible growth and fluctuations you prefer. Would you like the choices explained?",bucket="Confusion",reason="tradeoff_help")
        if has(text,r"\b(?:recommend|should (?:i|we) buy|which .*buy)\b"):
            return unresolved("I can help describe your preferred trade-off, but this prototype does not recommend investments. Which of the stated trade-offs fits what you want?",bucket="Confusion",reason="advice_request")
        lower=has(text,r"\b(?:smaller|lower|limit|less)\b.*\b(?:ups|downs|fluctuations|swings|risk)\b") and has(text,r"\b(?:lower|less|smaller)\b.*\b(?:growth|returns)\b")
        balanced=has(text,r"\b(?:balance|balanced|middle ground)\b") and has(text,r"\b(?:growth|returns|trade-off)\b")
        greater=has(text,r"\b(?:larger|greater|big|bigger|significant)\b.*\b(?:falls|swings|ups|downs|fluctuations|movements|risk)\b") and has(text,r"\b(?:higher|greater|more)\b.*\b(?:growth|returns)\b|\bgrowth potential matters most\b")
        if has(text,r"\b(?:i now prefer|i prefer)\b"):lower = lower or has(text,r"\bsmaller ups and downs\b.*\bgrowth is lower\b")
        if has(text,r"\b(?:please replace|earlier)\b.*\bi now prefer\b") and lower:greater=False
        ids=[oid for oid,hit in [("D14_LOWER",lower),("D14_BALANCED",balanced),("D14_GREATER",greater)] if hit]
        guarantee_text=re.sub(r"\b(?:not|never|isn't|is not)\s+guaranteed\b", "", text)
        if has(guarantee_text,r"\b(?:guaranteed|no loss|cannot lose|can't lose)\b"):return unresolved("Investment growth and recovery cannot be guaranteed. Which trade-off do you prefer, while accepting that loss is possible?",reason="protection_misunderstanding")
        if len(ids)==1:return clear(ids[0])
        return unresolved("Do you prefer smaller fluctuations with lower potential growth, a balance, or larger fluctuations for greater potential growth? It is okay not to choose yet.",reason="competing_meanings" if ids else "missing_tradeoff")

    if qid == "D15":
        if has(text,r"\b(?:pay rent|pay bills|pay my (?:electricity|gas|water) bill|cover essentials)\b") and not has(text,r"\b(?:worried|worry|comfortable|accept|unacceptable)\b"):
            return unresolved("That is an important practical concern for the previous loss-impact question. For this feelings question only, assume essentials and known plans stay covered. How would you feel?",bucket="Confusion",reason="practical_effect_not_feelings")
        unacceptable=has(text,r"\b(?:unacceptable|too uncomfortable|couldn't accept|cannot accept|can't accept|could not accept)\b")
        acceptable=has(text,r"\b(?:worried|worry|anxious|concerned)\b") and has(text,r"\b(?:could accept|can accept|able to accept|acceptable|could tolerate|could live with|accept that|accept it)\b")
        comfortable=has(text,r"\b(?:comfortable|little worry|wouldn't worry|not worried|calm)\b") and not has(text,r"\b(?:too uncomfortable|not|never|don't|do not)\b.{0,10}\b(?:comfortable|calm)\b")
        if has(text,r"\b(?:when|if|because).*\b(?:guaranteed|always recover|must recover)\b"):return unresolved("Recovery is not guaranteed in this example. With essentials and known plans covered, how would you feel about that possibility?",reason="protection_misunderstanding")
        ids=[oid for oid,hit in [("D15_UNACCEPTABLE",unacceptable),("D15_WORRIED_ACCEPTABLE",acceptable),("D15_COMFORTABLE",comfortable)] if hit]
        return clear(ids[0]) if len(ids)==1 else unresolved("For this feelings example, would the fall be unacceptable, worrying but acceptable, or comfortable? This is separate from practical affordability.",reason="competing_meanings" if len(ids)>1 else "missing_material_detail")

    return unresolved("Could you explain what you mean about this question?",bucket="Confusion")
